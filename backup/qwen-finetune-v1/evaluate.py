
"""Compare base Qwen3-0.6B against the LoRA adapter on Next.js questions.

Metrics (rough, automatic):
- accuracy: share of questions whose answer contains every expected keyword
- keyword_recall: average share of expected keywords found per answer
- relevance_f1: unigram F1 between the answer and the reference answer

Runs are batched, save after every batch, and resume from the results file
without repeating completed questions.
"""

import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==========================================
# 1. PATHS AND SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = ROOT / "models" / "nextjs-lora-test"
QUESTIONS_FILE = ROOT / "tests" / "questions.json"
RESULTS_DIR = ROOT / "tests" / "results"
RESULTS_FILE = RESULTS_DIR / "eval-nextjs.json"

MAX_NEW_TOKENS = 160
START_BATCH_SIZE = 5

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "you",
    "can", "use", "used", "which", "its", "into", "your", "they", "when",
    "such", "what", "does", "not", "all", "any", "has", "have", "will",
}


# ==========================================
# 2. SCORING
# ==========================================

def tokenize(text):
    words = re.findall(r"[a-z0-9@\[\]]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def relevance_f1(answer, reference):
    answer_tokens = set(tokenize(answer))
    reference_tokens = set(tokenize(reference))

    if not answer_tokens or not reference_tokens:
        return 0.0

    overlap = len(answer_tokens & reference_tokens)
    if overlap == 0:
        return 0.0

    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(answer, question):
    text = answer.lower()
    keywords = [k.lower() for k in question["keywords"]]
    found = [k for k in keywords if k in text]

    return {
        "correct": len(found) == len(keywords),
        "keyword_recall": len(found) / len(keywords),
        "relevance_f1": relevance_f1(answer, question["reference"]),
    }


def summarize(scored):
    total = len(scored)
    return {
        "accuracy": sum(s["correct"] for s in scored) / total,
        "keyword_recall": sum(s["keyword_recall"] for s in scored) / total,
        "relevance_f1": sum(s["relevance_f1"] for s in scored) / total,
    }


# ==========================================
# 3. RESULTS STORAGE (resume + safe writes)
# ==========================================

def load_state(questions):
    """Return saved state, or a fresh one. Refuses to mix incompatible runs."""
    config = {
        "base_model": BASE_MODEL,
        "adapter": str(ADAPTER_DIR),
        "max_new_tokens": MAX_NEW_TOKENS,
        "question_ids": [q["id"] for q in questions],
    }

    if RESULTS_FILE.exists():
        state = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        if state.get("config") != config:
            raise RuntimeError(
                f"{RESULTS_FILE} was made with different settings. "
                "Move it aside (do not delete) before starting a new run."
            )
        print(f"Resuming: {len(state['records'])} answers already saved.")
        return state

    return {"config": config, "records": [], "summary": None}


def save_state(state):
    """Write to a temp file first, then swap it in, so a crash never leaves a half-written file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = RESULTS_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, RESULTS_FILE)


def done_keys(state):
    return {(r["model"], r["id"]) for r in state["records"]}


# ==========================================
# 4. GENERATION
# ==========================================

def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_batch(model, tokenizer, items):
    prompts = [build_prompt(tokenizer, q["question"]) for q in items]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Left padding means every prompt ends at the same column.
    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
        for row in output
    ]


def gpu_status():
    alloc = torch.cuda.memory_allocated() / 2**20
    peak = torch.cuda.max_memory_allocated() / 2**20
    return f"VRAM now {alloc:.0f} MB, peak {peak:.0f} MB"


def run_pass(model, tokenizer, questions, state, model_name, use_base, batch_size, run_start):
    """Generate answers for every question not yet saved for this model."""
    finished = done_keys(state)
    pending = [q for q in questions if (model_name, q["id"]) not in finished]

    if not pending:
        print(f"[{model_name}] all questions already done, skipping.")
        return batch_size

    total = len(questions)
    done_before = total - len(pending)
    pass_start = time.time()
    done_here = 0
    i = 0

    print(f"\n[{model_name}] {len(pending)} questions to run, starting batch size {batch_size}")

    while i < len(pending):
        batch = pending[i:i + batch_size]
        context = model.disable_adapter() if use_base else nullcontext()

        try:
            with context:
                answers = generate_batch(model, tokenizer, batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"  CUDA out of memory, retrying with batch size {batch_size}")
            continue

        for item, answer in zip(batch, answers):
            scores = score_answer(answer, item)
            state["records"].append({
                "model": model_name,
                "id": item["id"],
                "question": item["question"],
                "answer": answer,
                **scores,
            })

        save_state(state)

        i += len(batch)
        done_here += len(batch)
        elapsed = time.time() - run_start
        per_question = (time.time() - pass_start) / done_here
        left = len(pending) - i
        print(
            f"  [{model_name}] {done_before + done_here}/{total} done | "
            f"elapsed {elapsed / 60:.1f} min | {per_question:.1f} s/question | "
            f"{left} left in this pass | batch {batch_size} | {gpu_status()}"
        )

    return batch_size


# ==========================================
# 5. MAIN
# ==========================================

def summary_for(state, model_name):
    scored = [r for r in state["records"] if r["model"] == model_name]
    return summarize(scored) if scored else None


def main():
    run_start = time.time()
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    state = load_state(questions)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Qwen3-0.6B once...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).eval()
    print(f"Loaded. {gpu_status()}")

    batch_size = START_BATCH_SIZE
    batch_size = run_pass(model, tokenizer, questions, state, "base", True, batch_size, run_start)
    batch_size = run_pass(model, tokenizer, questions, state, "adapter", False, batch_size, run_start)

    base_summary = summary_for(state, "base")
    adapter_summary = summary_for(state, "adapter")
    state["summary"] = {"base": base_summary, "adapter": adapter_summary}
    save_state(state)

    print("\n================================")
    print("RESULTS")
    print("================================")
    print(f"{'metric':<16}{'base':>10}{'adapter':>10}{'diff':>10}")
    for metric in ["accuracy", "keyword_recall", "relevance_f1"]:
        b = base_summary[metric]
        a = adapter_summary[metric]
        print(f"{metric:<16}{b:>10.3f}{a:>10.3f}{a - b:>+10.3f}")

    by_key = {(r["model"], r["id"]): r for r in state["records"]}
    print("\nPer-question:")
    for item in questions:
        b = by_key[("base", item["id"])]
        a = by_key[("adapter", item["id"])]
        print(
            f"Q{item['id']:>2} base={'OK ' if b['correct'] else 'no '} "
            f"adapter={'OK ' if a['correct'] else 'no '} "
            f"f1 base={b['relevance_f1']:.2f} adapter={a['relevance_f1']:.2f}  "
            f"{item['question']}"
        )

    print(f"\nTotal time: {(time.time() - run_start) / 60:.1f} min. Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
