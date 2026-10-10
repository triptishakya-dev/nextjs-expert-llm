"""Train the custom BPE tokenizer on the training Q&A text.

Only the training split is used, so validation text does not shape the vocabulary.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.bpe import BPETokenizer  # noqa: E402
from llm.dataset import load_jsonl, tokenizer_training_texts  # noqa: E402

TRAIN_FILE = ROOT / "data" / "qa" / "train.jsonl"
VAL_FILE = ROOT / "data" / "qa" / "validation.jsonl"
OUT_FILE = ROOT / "models" / "gpt" / "tokenizer.json"

VOCAB_SIZE = 8000


def main():
    start = time.time()
    tokenizer = BPETokenizer.train(tokenizer_training_texts(TRAIN_FILE), VOCAB_SIZE)
    tokenizer.save(OUT_FILE)
    print(f"Trained tokenizer: vocab size {tokenizer.vocab_size} "
          f"({len(tokenizer.merges)} merges) in {time.time() - start:.1f}s -> {OUT_FILE}")

    # Compression check on held-out validation text
    val_texts = [r["question"] + " " + r["answer"] for r in load_jsonl(VAL_FILE)]
    chars = sum(len(t) for t in val_texts)
    tokens = sum(len(tokenizer.encode(t)) for t in val_texts)
    print(f"Validation: {chars} characters -> {tokens} tokens "
          f"({chars / tokens:.2f} characters per token)")


if __name__ == "__main__":
    main()
