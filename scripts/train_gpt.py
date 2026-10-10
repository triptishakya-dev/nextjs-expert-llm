"""Two-stage training for the from-scratch GPT.

Stage 1: pretrain   randomly initialized weights, next-token prediction on raw Next.js docs
Stage 2: instruct   continue from the stage 1 checkpoint, on the Q&A data (loss on answers only)

Each stage writes its own folder with its own checkpoints:

    models/gpt/pretrain/best.pt, last.pt, metrics.jsonl, summary.json
    models/gpt/instruct/best.pt, last.pt, metrics.jsonl, summary.json

Both stages log training loss and validation loss. When a stage ends, the
20-question keyword evaluation runs on its best checkpoint, and the results go
to tests/results/gpt-<stage>.json (skip with --skip-eval).

Smoke test first (both stages, a few steps each):
    python scripts/train_gpt.py --stage pretrain --max-steps 3 --eval-every 2 --device cpu --out-dir <scratch>
    python scripts/train_gpt.py --stage instruct --max-steps 3 --eval-every 2 --device cpu --out-dir <scratch>
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.bpe import BPETokenizer, EOS_ID, BOS_ID  # noqa: E402
from llm.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from llm.dataset import PackedDataset, QADataset, collate, format_prompt  # noqa: E402
from llm.model import GPT, GPTConfig  # noqa: E402

TOKENIZER_FILE = ROOT / "models" / "gpt" / "tokenizer.json"
MODELS_DIR = ROOT / "models" / "gpt"
STAGE_DEFAULTS = {
    # stage: (data dir, max steps, peak lr, min lr, warmup steps)
    "pretrain": (ROOT / "data" / "pretrain", 2000, 6e-4, 6e-5, 100),
    "instruct": (ROOT / "data" / "qa", 1500, 3e-4, 3e-5, 50),
}
SAMPLE_QUESTION = "How do you create a dynamic route segment in the Next.js App Router?"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pretrain", "instruct"], required=True)
    p.add_argument("--init-from", type=Path, default=None,
                   help="Checkpoint to continue from. Required for instruct; "
                        "defaults to models/gpt/pretrain/best.pt.")
    p.add_argument("--data-dir", type=Path, default=None, help="Defaults to the stage's data folder.")
    p.add_argument("--out-dir", type=Path, default=None, help="Defaults to models/gpt/<stage>.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=512, help="Pretraining sequence length.")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--skip-eval", action="store_true", help="Do not run the 20-question evaluation at the end.")
    args = p.parse_args()

    data_dir, steps, lr, min_lr, warmup = STAGE_DEFAULTS[args.stage]
    args.data_dir = args.data_dir or data_dir
    args.max_steps = args.max_steps if args.max_steps is not None else steps
    args.lr = args.lr if args.lr is not None else lr
    args.min_lr = args.min_lr if args.min_lr is not None else min_lr
    args.warmup_steps = args.warmup_steps if args.warmup_steps is not None else warmup
    args.out_dir = args.out_dir or MODELS_DIR / args.stage
    if args.stage == "instruct" and args.init_from is None:
        args.init_from = MODELS_DIR / "pretrain" / "best.pt"
    return args


def lr_at(step, args):
    """Linear warmup, then cosine decay from lr down to min_lr."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return args.min_lr + (args.lr - args.min_lr) * cosine


def make_optimizer(model, args):
    # Weight decay on weight matrices only. Biases and LayerNorm weights are left alone.
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )


def make_datasets(args, tokenizer, cfg):
    if args.stage == "pretrain":
        block = min(args.block_size, cfg.context_length)
        train = PackedDataset(args.data_dir / "train.jsonl", tokenizer, block)
        val = PackedDataset(args.data_dir / "validation.jsonl", tokenizer, block)
    else:
        train = QADataset(args.data_dir / "train.jsonl", tokenizer, cfg.context_length)
        val = QADataset(args.data_dir / "validation.jsonl", tokenizer, cfg.context_length)
    return train, val


@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        total += loss.item()
        count += 1
    model.train()
    return total / max(1, count)


@torch.no_grad()
def sample_answer(model, tokenizer, device, max_new_tokens=80):
    model.eval()
    prompt = [BOS_ID] + tokenizer.encode(format_prompt(SAMPLE_QUESTION))
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40, eos_id=EOS_ID)
    model.train()
    text = tokenizer.decode(out[0, len(prompt):].tolist()).strip()[:200]
    # Escape anything the console cannot show (early samples are often random bytes)
    return text.encode("ascii", "backslashreplace").decode("ascii")


def run_keyword_eval(stage, checkpoint, out_dir):
    """Run the 20-question evaluation on the best checkpoint, in its own process.

    The full results (answers and scores) are saved in the stage folder. Copy
    the file into tests/results/ when it is a run you want to keep.
    """
    output = out_dir / "keyword_eval.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate_gpt.py"),
           "--checkpoint", str(checkpoint), "--name", stage, "--output", str(output)]
    subprocess.run(cmd, check=True)
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    tokenizer = BPETokenizer.load(TOKENIZER_FILE)

    if args.stage == "pretrain":
        cfg = GPTConfig(vocab_size=tokenizer.vocab_size)
        model = GPT(cfg).to(device)
        origin = "random initialization"
    else:
        model, meta = load_checkpoint(args.init_from, device=device)
        cfg = GPTConfig(**meta["config"])
        model.train()
        origin = f"{args.init_from} (stage {meta.get('stage')}, step {meta.get('step')})"

    print(f"Stage: {args.stage} | init: {origin}")
    print(f"Device: {device} | parameters: {model.num_parameters() / 1e6:.2f}M | fp16: {use_amp}")

    train_set, val_set = make_datasets(args, tokenizer, cfg)
    print(f"Train examples: {len(train_set)} | validation examples: {len(val_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")  # start a fresh log for this run

    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, best_step = float("inf"), 0
    step, window_loss, window_count = 0, 0.0, 0
    start = time.time()
    model.train()

    def log(row):
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    while step < args.max_steps:
        for x, y in train_loader:
            if step >= args.max_steps:
                break

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args)

            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            window_loss += loss.item()
            window_count += 1

            if step % args.log_every == 0:
                train_loss = window_loss / window_count
                print(f"step {step:5d} | train loss {train_loss:.4f} | lr {lr_at(step, args):.2e} "
                      f"| {time.time() - start:.0f}s")
                log({"step": step, "train_loss": train_loss, "lr": lr_at(step, args),
                     "elapsed_s": round(time.time() - start, 1)})
                window_loss, window_count = 0.0, 0

            if step % args.eval_every == 0 or step == args.max_steps:
                val_loss = evaluate_loss(model, val_loader, device, use_amp)
                print(f"step {step:5d} | VALIDATION loss {val_loss:.4f}")
                print(f"  sample: {sample_answer(model, tokenizer, device)!r}")
                log({"step": step, "val_loss": val_loss, "elapsed_s": round(time.time() - start, 1)})
                if val_loss < best_val:
                    best_val, best_step = val_loss, step
                    save_checkpoint(args.out_dir / "best.pt", model, cfg, args.stage, step,
                                    {"val_loss": val_loss})
                    print(f"  saved best checkpoint -> {args.out_dir / 'best.pt'}")

    save_checkpoint(args.out_dir / "last.pt", model, cfg, args.stage, step, {"val_loss": best_val})

    summary = {
        "stage": args.stage,
        "init_from": str(args.init_from) if args.init_from else None,
        "parameters": model.num_parameters(),
        "steps": step,
        "best_val_loss": best_val,
        "best_step": best_step,
        "minutes": round((time.time() - start) / 60, 2),
        "config": dict(cfg.__dict__),
        "hyperparameters": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))},
    }

    if not args.skip_eval:
        summary["keyword_eval"] = run_keyword_eval(args.stage, args.out_dir / "best.pt", args.out_dir)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Best validation loss {best_val:.4f} at step {best_step}. "
          f"Total time {(time.time() - start) / 60:.1f} min. Summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
