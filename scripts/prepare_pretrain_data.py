"""Prepare raw Next.js docs for stage 1 (next-token pretraining).

Every .mdx page under data/raw is cleaned the same way as the Q&A data.
Evaluation source pages are excluded (see llm/eval_pages.py). The split is by
page, so no page appears in both training and validation.

Output: data/pretrain/train.jsonl and data/pretrain/validation.jsonl, one
{"text", "source"} record per page. Tokenizing and packing happen in training.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_dataset import clean_mdx  # noqa: E402
from generate_qa import read_frontmatter, strip_jsx  # noqa: E402
from llm.eval_pages import is_eval_source  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "pretrain"
TRAIN_OUT = OUT_DIR / "train.jsonl"
VAL_OUT = OUT_DIR / "validation.jsonl"

MIN_PAGE_CHARS = 100
VAL_FRACTION = 0.1
SEED = 42


def page_text(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta = read_frontmatter(raw)
    title = meta.get("title", path.stem)
    body = strip_jsx(clean_mdx(raw))
    return f"# {title}\n\n{body}"


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    all_pages = sorted(RAW_DIR.rglob("*.mdx"))
    pages = [p for p in all_pages if not is_eval_source(p)]
    excluded = len(all_pages) - len(pages)

    records = []
    for path in pages:
        text = page_text(path)
        if len(text) < MIN_PAGE_CHARS:
            continue
        records.append({"text": text, "source": path.relative_to(RAW_DIR).as_posix()})

    if not records:
        raise RuntimeError("No pages left for pretraining. Run collect_docs.py first.")

    random.Random(SEED).shuffle(records)
    split = max(1, int(len(records) * (1 - VAL_FRACTION)))
    train, val = records[:split], records[split:]

    # Safety check: no eval source page may be in the output
    for record in train + val:
        assert not is_eval_source(Path(record["source"])), record["source"]

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VAL_OUT, val)

    train_chars = sum(len(r["text"]) for r in train)
    val_chars = sum(len(r["text"]) for r in val)
    print(f"Pages: {len(all_pages)} total, {excluded} excluded as eval sources")
    print(f"Pretraining pages: {len(records)} (train {len(train)}, validation {len(val)})")
    print(f"Characters: train {train_chars:,}, validation {val_chars:,}")
    print(f"Saved: {TRAIN_OUT}, {VAL_OUT}")


if __name__ == "__main__":
    main()
