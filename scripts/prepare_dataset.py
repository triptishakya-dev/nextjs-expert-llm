
import json
import random
import re
from pathlib import Path

from tqdm import tqdm


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TRAIN_FILE = ROOT / "data" / "train.jsonl"
VAL_FILE = ROOT / "data" / "validation.jsonl"

CHUNK_SIZE = 1800

random.seed(42)


# ==========================================
# 2. CLEAN MDX DOCUMENTATION
# ==========================================

def clean_mdx(content):

    # Remove YAML frontmatter
    content = re.sub(
        r"\A---\s*\n.*?\n---\s*\n",
        "",
        content,
        flags=re.DOTALL,
    )

    # Remove JSX comments
    content = re.sub(
        r"\{/\*.*?\*/\}",
        "",
        content,
        flags=re.DOTALL,
    )

    # Remove HTML comments
    content = re.sub(
        r"<!--.*?-->",
        "",
        content,
        flags=re.DOTALL,
    )

    # Remove simple import statements
    content = re.sub(
        r"^import\s+.*$",
        "",
        content,
        flags=re.MULTILINE,
    )

    # Normalize extra empty lines
    content = re.sub(
        r"\n{3,}",
        "\n\n",
        content,
    )

    return content.strip()


# ==========================================
# 3. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_text(text, max_chars=CHUNK_SIZE):

    paragraphs = text.split("\n\n")

    chunks = []
    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Split paragraphs larger than max_chars
        pieces = [
            paragraph[i:i + max_chars]
            for i in range(0, len(paragraph), max_chars)
        ]

        for piece in pieces:

            if len(current) + len(piece) + 2 <= max_chars:
                current += piece + "\n\n"

            else:
                if current.strip():
                    chunks.append(current.strip())

                current = piece + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ==========================================
# 4. SAVE JSONL FILE
# ==========================================

def write_jsonl(path, records):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


# ==========================================
# 5. PROCESS DOCUMENTATION FILES
# ==========================================

def process_files(source_files):

    records = []

    for file in tqdm(source_files):

        content = file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        cleaned = clean_mdx(content)

        # Skip empty or placeholder documents
        if len(cleaned) < 100:
            continue

        relative = file.relative_to(RAW_DIR)

        # Save cleaned documentation
        output_path = (
            PROCESSED_DIR / relative
        ).with_suffix(".txt")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            cleaned,
            encoding="utf-8",
        )

        # Create training chunks
        chunks = split_text(cleaned)

        for chunk in chunks:

            if len(chunk) < 100:
                continue

            records.append({
                "text": chunk,
                "source": relative.as_posix(),
            })

    return records


# ==========================================
# 6. PREPARE TRAIN AND VALIDATION DATA
# ==========================================

def prepare_dataset():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_DIR.rglob("*.mdx")
    )

    if not files:
        raise RuntimeError(
            "No MDX files found. Run collect_docs.py first."
        )

    print(f"Found {len(files)} documentation files")

    # Split by document to avoid sharing
    # chunks from the same document across sets
    random.shuffle(files)

    split_index = max(
        1,
        int(len(files) * 0.9)
    )

    train_files = files[:split_index]
    val_files = files[split_index:]

    print("\nProcessing training documentation...")
    train_data = process_files(train_files)

    print("\nProcessing validation documentation...")
    val_data = process_files(val_files)

    if not train_data or not val_data:
        raise RuntimeError(
            "Dataset is empty. Check your MDX files."
        )

    write_jsonl(
        TRAIN_FILE,
        train_data,
    )

    write_jsonl(
        VAL_FILE,
        val_data,
    )

    print("\n================================")
    print("DATASET PREPARATION COMPLETED")
    print("================================")

    print("Training examples:", len(train_data))
    print("Validation examples:", len(val_data))

    print("\nFiles saved:")
    print(TRAIN_FILE)
    print(VAL_FILE)


# ==========================================
# 7. RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    prepare_dataset()
