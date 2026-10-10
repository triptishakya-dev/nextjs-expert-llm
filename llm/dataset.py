"""Turn Q&A records into token ids the GPT model can train on.

One example looks like this, as token ids:

    <bos> Question: How do you ...?\nAnswer: <answer tokens> <eos>

Loss is only computed on the answer tokens (and the final <eos>). The question
is still in the input, but its positions have label -100, which the model's
loss ignores. So the model is trained to answer, not to repeat questions.
"""

import json

import torch
from torch.utils.data import Dataset

from llm.bpe import BOS_ID, EOS_ID, PAD_ID

IGNORE_INDEX = -100


def format_prompt(question):
    return f"Question: {question}\nAnswer:"


def encode_record(tokenizer, record):
    """Return (prompt_ids, answer_ids) for one record.

    The leading space before the answer belongs to the answer, so the prompt
    ends cleanly at "Answer:".
    """
    prompt_ids = [BOS_ID] + tokenizer.encode(format_prompt(record["question"]))
    answer_ids = tokenizer.encode(" " + record["answer"]) + [EOS_ID]
    return prompt_ids, answer_ids


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class QADataset(Dataset):
    """Pre-encodes every record. Examples longer than the context are skipped."""

    def __init__(self, path, tokenizer, context_length):
        self.examples = []
        self.skipped = 0

        for record in load_jsonl(path):
            prompt_ids, answer_ids = encode_record(tokenizer, record)
            ids = prompt_ids + answer_ids
            if len(ids) > context_length + 1:
                self.skipped += 1
                continue

            # Input is every token but the last; the target is every token but the first.
            x = ids[:-1]
            y = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[1:]
            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class PackedDataset(Dataset):
    """Stage 1 data: every page is tokenized, pages are joined with <eos>,
    and the whole stream is cut into blocks of block_size + 1 tokens.

    Each block gives an input (the first block_size tokens) and a target (the
    same tokens shifted by one), so the model learns to predict every next token.
    """

    def __init__(self, path, tokenizer, block_size):
        stream = []
        for record in load_jsonl(path):
            stream.extend(tokenizer.encode(record["text"]))
            stream.append(EOS_ID)

        self.total_tokens = len(stream)
        self.blocks = []
        for start in range(0, len(stream) - block_size, block_size):
            self.blocks.append(stream[start:start + block_size + 1])

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        block = self.blocks[i]
        return block[:-1], block[1:]


def collate(batch):
    """Pad a list of (x, y) pairs to the same length within the batch."""
    max_len = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = max_len - len(x)
        xs.append(x + [PAD_ID] * pad)
        ys.append(y + [IGNORE_INDEX] * pad)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def tokenizer_training_texts(path):
    """Plain text for tokenizer training, in the same shape the model will see."""
    for record in load_jsonl(path):
        yield format_prompt(record["question"]) + " " + record["answer"]
