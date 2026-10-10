"""Turn the Next.js App Router docs into instruction-style Q&A pairs.

Answers are copied from the docs themselves (no model writes them), so every
answer is accurate to the source. Questions are built from the page title and
section headings.

Pages that the evaluation questions are based on are excluded, so the
evaluation measures what the model learned from other pages, not memorized
answers. Output is split by document, so pairs from one page never appear in
both the training and validation sets.

Each record is plain text. The model's own tokenizer and template are applied
later, in llm/dataset.py:

    {"question": "...", "answer": "...", "source": "02-guides/..."}
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_dataset import clean_mdx  # noqa: E402
from llm.eval_pages import is_eval_source  # noqa: E402


# ==========================================
# 1. PATHS AND SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "01-app"  # App Router docs only
OUT_DIR = ROOT / "data" / "qa"
TRAIN_OUT = OUT_DIR / "train.jsonl"
VAL_OUT = OUT_DIR / "validation.jsonl"

MIN_ANSWER_CHARS = 150
MAX_ANSWER_CHARS = 1200
MIN_DOC_DESCRIPTION_CHARS = 40
VAL_FRACTION = 0.1

random.seed(42)


# ==========================================
# 2. CLEANING
# ==========================================

def strip_jsx(text):
    """Remove JSX wrappers and PagesOnly content, keep the prose and code."""
    # Pages Router only content does not apply to the App Router docs
    text = re.sub(r"<PagesOnly>.*?</PagesOnly>", "", text, flags=re.DOTALL)
    # Lines that hold only a JSX tag such as <AppOnly> or </Note>
    text = re.sub(r"^\s*</?[A-Z][A-Za-z0-9]*[^>]*>\s*$", "", text, flags=re.MULTILINE)
    # Inline JSX tags left inside a sentence
    text = re.sub(r"</?[A-Z][A-Za-z0-9]*[^>]*>", "", text)
    # Markdown links [text](/docs/...) -> text
    text = re.sub(r"\[([^\]]+)\]\((?!https?://)[^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]*\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_frontmatter(raw):
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", raw, flags=re.DOTALL)
    meta = {}
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def truncate(text, max_chars=MAX_ANSWER_CHARS):
    """Cut at a paragraph break so the answer does not end mid-sentence."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_break = cut.rfind("\n\n")
    if last_break > MIN_ANSWER_CHARS:
        return cut[:last_break].strip()
    return cut.rsplit(" ", 1)[0].strip()


# ==========================================
# 3. SPLIT A PAGE INTO SECTIONS
# ==========================================

def split_sections(text):
    """Return (heading, body) pairs for every ## and ### heading.

    Headings inside fenced code blocks are ignored.
    """
    sections = []
    current_heading = None
    current_lines = []
    in_fence = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence

        heading_match = None if in_fence else re.match(r"^(##|###) (.+?)\s*$", line)

        if heading_match:
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections


def clean_heading(heading):
    return heading.replace("`", "").strip()


# ==========================================
# 4. BUILD Q&A PAIRS FROM ONE PAGE
# ==========================================

def pairs_from_page(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta = read_frontmatter(raw)
    title = meta.get("title", path.stem.replace("-", " ").title())
    description = meta.get("description", "")

    cleaned = strip_jsx(clean_mdx(raw))
    pairs = []

    # Page-level question, answered by the page description
    if len(description) >= MIN_DOC_DESCRIPTION_CHARS:
        pairs.append({
            "question": f"What does the Next.js '{title}' documentation page cover?",
            "answer": description,
        })

    # Section-level questions, answered by the section body
    for heading, body in split_sections(cleaned):
        if len(body) < MIN_ANSWER_CHARS:
            continue
        heading_text = clean_heading(heading)
        pairs.append({
            "question": f"In Next.js, how does '{heading_text}' work in the '{title}' docs?",
            "answer": truncate(body),
        })

    return pairs


# ==========================================
# 5. WRITE AND MAIN
# ==========================================

def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    all_files = sorted(RAW_DIR.rglob("*.mdx"))
    files = [p for p in all_files if not is_eval_source(p)]
    excluded = len(all_files) - len(files)

    if not files:
        raise RuntimeError("No App Router MDX files left after exclusions. Run collect_docs.py first.")

    # Split by document so pairs from one page never land in both sets
    random.shuffle(files)
    split_index = max(1, int(len(files) * (1 - VAL_FRACTION)))
    train_files = files[:split_index]
    val_files = files[split_index:]

    def build(file_list):
        records = []
        seen_questions = set()
        for path in file_list:
            source = path.relative_to(RAW_DIR).as_posix()
            for pair in pairs_from_page(path):
                if pair["question"] in seen_questions:
                    continue
                seen_questions.add(pair["question"])
                records.append({
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "source": source,
                })
        return records

    train_records = build(train_files)
    val_records = build(val_files)

    if not train_records or not val_records:
        raise RuntimeError("Q&A dataset is empty. Check the source files.")

    write_jsonl(TRAIN_OUT, train_records)
    write_jsonl(VAL_OUT, val_records)

    print(f"Pages: {len(all_files)} total, {excluded} excluded as eval sources")
    print(f"Pages used: {len(files)} (train {len(train_files)}, validation {len(val_files)})")
    print(f"Train Q&A pairs: {len(train_records)} -> {TRAIN_OUT}")
    print(f"Validation Q&A pairs: {len(val_records)} -> {VAL_OUT}")


if __name__ == "__main__":
    main()
