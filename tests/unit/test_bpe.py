import tempfile
import unittest
from pathlib import Path

from llm.bpe import BOS_ID, EOS_ID, PAD_ID, UNK_ID, BPETokenizer

CORPUS = [
    "The App Router uses folders for routes. " * 20,
    "Use the useRouter hook in a Client Component. " * 20,
    "Dynamic routes use square brackets, like [slug]. " * 20,
]


class BPETokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer.train(CORPUS, vocab_size=300)

    def test_vocab_size_matches_request_or_less(self):
        self.assertLessEqual(self.tok.vocab_size, 300)
        self.assertGreater(len(self.tok.merges), 0)

    def test_round_trip_plain_text(self):
        text = "Use useRouter in a Client Component."
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_round_trip_unicode_and_unseen_text(self):
        # Byte-level: characters never seen in training still encode and decode
        text = "naïve café — 日本語 🚀"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_round_trip_code_with_whitespace(self):
        text = "import { useRouter } from 'next/navigation';\n\n  const x = 1;\n"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_empty_string(self):
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(self.tok.decode([]), "")

    def test_special_tokens_are_never_produced_by_encode(self):
        ids = self.tok.encode("<bos> <eos> <pad> " * 5)
        self.assertFalse({BOS_ID, EOS_ID, PAD_ID, UNK_ID} & set(ids))

    def test_special_ids_are_dropped_by_decode(self):
        ids = [BOS_ID] + self.tok.encode("hello") + [EOS_ID, PAD_ID]
        self.assertEqual(self.tok.decode(ids), "hello")

    def test_frequent_text_compresses(self):
        text = "The App Router uses folders for routes. "
        self.assertLess(len(self.tok.encode(text)), len(text.encode("utf-8")))

    def test_training_is_deterministic(self):
        again = BPETokenizer.train(CORPUS, vocab_size=300)
        self.assertEqual(again.merges, self.tok.merges)

    def test_save_and_load_give_same_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            self.tok.save(path)
            loaded = BPETokenizer.load(path)
        text = "Dynamic routes use square brackets."
        self.assertEqual(loaded.encode(text), self.tok.encode(text))

    def test_vocab_too_small_is_rejected(self):
        with self.assertRaises(ValueError):
            BPETokenizer.train(CORPUS, vocab_size=100)


if __name__ == "__main__":
    unittest.main()
