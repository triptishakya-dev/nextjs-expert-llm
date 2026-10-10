"""Evaluate a custom GPT checkpoint on the 20 held-out questions.

Uses the same questions, keyword scoring and metrics as the Qwen-era
scripts/evaluate.py, and the custom BPE tokenizer in models/gpt/tokenizer.json.
Results are written to JSON so stages and runs can be compared later.

Examples:
    python scripts/evaluate_gpt.py --checkpoint models/gpt/instruct/best.pt
    python scripts/evaluate_gpt.py --random-init --name random-init
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.bpe import BPETokenizer  # noqa: E402
from llm.checkpoint import load_checkpoint  # noqa: E402
from llm.evaluation import evaluate_model  # noqa: E402
from llm.model import GPT, GPTConfig  # noqa: E402

TOKENIZER_FILE = ROOT / "models" / "gpt" / "tokenizer.json"
QUESTIONS_FILE = ROOT / "tests" / "questions.json"
RESULTS_DIR = ROOT / "tests" / "results"


def parse_args():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Path to a trained .pt checkpoint.")
    source.add_argument("--random-init", action="store_true",
                        help="Evaluate an untrained model with the default config (baseline).")
    p.add_argument("--name", help="Label for the results file. Defaults to the checkpoint's stage.")
    p.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--output", type=Path, help="JSON output path. Defaults to tests/results/gpt-<name>.json.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = BPETokenizer.load(TOKENIZER_FILE)

    if args.random_init:
        torch.manual_seed(0)
        cfg = GPTConfig(vocab_size=tokenizer.vocab_size)
        model = GPT(cfg).to(device).eval()
        meta = {"stage": "random-init", "step": 0, "config": dict(cfg.__dict__)}
        name = args.name or "random-init"
        source = "random initialization (seed 0)"
    else:
        model, meta = load_checkpoint(args.checkpoint, device=device)
        name = args.name or f"{meta.get('stage', 'checkpoint')}"
        source = str(args.checkpoint)

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    print(f"Evaluating {name} ({source}) on {len(questions)} questions...")

    start = time.time()
    records, summary = evaluate_model(model, tokenizer, questions, args.max_new_tokens, device)
    elapsed = time.time() - start

    result = {
        "name": name,
        "checkpoint": source,
        "stage": meta.get("stage"),
        "step": meta.get("step"),
        "parameters": model.num_parameters(),
        "config": meta.get("config"),
        "tokenizer": str(TOKENIZER_FILE.relative_to(ROOT).as_posix()),
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "questions_file": str(args.questions.name),
        "num_questions": len(questions),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
        "records": records,
    }

    output = args.output or RESULTS_DIR / f"gpt-{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'metric':<16}{name:>16}")
    for metric in ["accuracy", "keyword_recall", "relevance_f1"]:
        print(f"{metric:<16}{summary[metric]:>16.3f}")
    print(f"\nAnswers generated in {elapsed:.1f}s. Saved to {output}")


if __name__ == "__main__":
    main()
