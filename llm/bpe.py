"""A byte-level BPE tokenizer, written from scratch.

How it works, in short:
1. Text is cut into "words" with a regex (a word, a number, a run of
   punctuation, or whitespace). Merges never cross these boundaries.
2. Each word is turned into UTF-8 bytes. Every byte is a starting token,
   so any text can be encoded and nothing is ever "unknown".
3. Training counts how often each neighbouring pair of tokens appears, merges
   the most frequent pair into a new token, and repeats until the vocabulary
   reaches the target size. The ordered list of merges is the tokenizer.
4. Encoding applies the merges in the same order they were learned.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Words: optional contraction, optional space + letters, optional space +
# digits, optional space + punctuation, or whitespace.
WORD_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[A-Za-z]+"
    r"| ?[0-9]+"
    r"| ?[^\sA-Za-z0-9]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)
BYTE_OFFSET = len(SPECIAL_TOKENS)  # byte b is token id b + BYTE_OFFSET


class BPETokenizer:
    def __init__(self, merges=None):
        # merges: list of (left_id, right_id) pairs, in the order they were learned.
        # The new token for merge i is BYTE_OFFSET + 256 + i.
        self.merges = [tuple(m) for m in (merges or [])]
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = self._build_vocab()

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def _build_vocab(self):
        """Map every token id to its bytes. Special tokens have no bytes."""
        vocab = {}
        for byte in range(256):
            vocab[byte + BYTE_OFFSET] = bytes([byte])
        for i, (left, right) in enumerate(self.merges):
            vocab[BYTE_OFFSET + 256 + i] = vocab[left] + vocab[right]
        return vocab

    @property
    def vocab_size(self):
        return BYTE_OFFSET + 256 + len(self.merges)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(cls, texts, vocab_size, min_pair_count=2):
        """Learn merges from an iterable of strings until vocab_size is reached."""
        target_merges = vocab_size - BYTE_OFFSET - 256
        if target_merges < 0:
            raise ValueError(f"vocab_size must be at least {BYTE_OFFSET + 256}")

        # Count each distinct word once, with its frequency
        word_freq = Counter()
        for text in texts:
            for word in WORD_PATTERN.findall(text):
                word_freq[word.encode("utf-8")] += 1

        words = [[b + BYTE_OFFSET for b in word] for word in word_freq]
        freqs = list(word_freq.values())

        # Pair counts over all words, plus which words contain each pair
        pair_counts = Counter()
        pair_where = defaultdict(set)
        for i, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[i]
                pair_where[pair].add(i)

        merges = []
        for _ in range(target_merges):
            if not pair_counts:
                break

            # Most frequent pair. Ties are broken by the pair itself, so training is repeatable.
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best] < min_pair_count:
                break

            new_id = BYTE_OFFSET + 256 + len(merges)
            merges.append(best)

            # Rewrite only the words that contain the pair, and update counts incrementally
            for i in list(pair_where[best]):
                word, freq = words[i], freqs[i]

                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]

                merged = _merge_word(word, best, new_id)
                words[i] = merged

                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += freq
                    pair_where[pair].add(i)

            pair_where.pop(best, None)

        return cls(merges)

    # ------------------------------------------------------------------
    # Encoding and decoding
    # ------------------------------------------------------------------

    def encode(self, text):
        ids = []
        for word in WORD_PATTERN.findall(text):
            ids.extend(self._encode_word(word.encode("utf-8")))
        return ids

    def _encode_word(self, word_bytes):
        ids = [b + BYTE_OFFSET for b in word_bytes]

        while len(ids) > 1:
            # Find the adjacent pair that was learned earliest
            ranked = [
                (self.merge_rank[pair], pair)
                for pair in zip(ids, ids[1:])
                if pair in self.merge_rank
            ]
            if not ranked:
                break
            _, pair = min(ranked)
            new_id = BYTE_OFFSET + 256 + self.merge_rank[pair]
            ids = _merge_word(ids, pair, new_id)

        return ids

    def decode(self, ids):
        """Turn ids back into text. Special tokens are dropped."""
        data = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "byte-level-bpe",
            "special_tokens": SPECIAL_TOKENS,
            "merges": [list(m) for m in self.merges],
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["special_tokens"] != SPECIAL_TOKENS:
            raise ValueError("Tokenizer file was made with different special tokens.")
        return cls(payload["merges"])


def _merge_word(word, pair, new_id):
    """Replace every non-overlapping occurrence of pair in word with new_id."""
    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out
