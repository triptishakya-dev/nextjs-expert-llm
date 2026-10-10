import json
import tempfile
import unittest
from pathlib import Path

import torch

from llm.bpe import BPETokenizer
from llm.checkpoint import load_checkpoint, save_checkpoint
from llm.dataset import PackedDataset
from llm.eval_pages import is_eval_source
from llm.evaluation import evaluate_model, relevance_f1, score_answer, summarize
from llm.model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[2]
TINY = GPTConfig(vocab_size=300, context_length=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


class ScoringTests(unittest.TestCase):
    def test_keyword_accuracy_requires_every_keyword(self):
        question = {"keywords": ["[", "]"], "reference": "Use square brackets for dynamic segments."}
        self.assertTrue(score_answer("Use app/blog/[slug] folders.", question)["correct"])
        self.assertFalse(score_answer("Use app/blog/slug folders.", question)["correct"])

    def test_keyword_recall_is_fraction_found(self):
        question = {"keywords": ["layout", "page"], "reference": "x"}
        scores = score_answer("A layout wraps the segment.", question)
        self.assertEqual(scores["keyword_recall"], 0.5)
        self.assertFalse(scores["correct"])

    def test_relevance_f1_bounds(self):
        self.assertEqual(relevance_f1("", "anything useful"), 0.0)
        self.assertAlmostEqual(relevance_f1("layout wraps segments", "layout wraps segments"), 1.0)
        self.assertEqual(relevance_f1("cats dogs", "layout wraps segments"), 0.0)

    def test_summary_averages(self):
        scored = [
            {"correct": True, "keyword_recall": 1.0, "relevance_f1": 1.0},
            {"correct": False, "keyword_recall": 0.0, "relevance_f1": 0.0},
        ]
        summary = summarize(scored)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["keyword_recall"], 0.5)


class PackedDatasetTests(unittest.TestCase):
    def test_blocks_are_shifted_by_one_and_separated_by_eos(self):
        tok = BPETokenizer.train(["hello world, this is a test. " * 40], vocab_size=300)
        records = [{"text": "hello world " * 20}, {"text": "this is a test " * 20}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pretrain.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            ds = PackedDataset(path, tok, block_size=16)

        self.assertGreater(len(ds), 0)
        x, y = ds[0]
        self.assertEqual(len(x), 16)
        self.assertEqual(len(y), 16)
        # Target is the input shifted left by one position
        self.assertEqual(x[1:], y[:-1])
        # The stream is every page's tokens plus one <eos> per page
        expected_stream = sum(len(tok.encode(r["text"])) + 1 for r in records)
        self.assertEqual(ds.total_tokens, expected_stream)


class EvalExclusionTests(unittest.TestCase):
    def test_eval_pages_are_detected(self):
        self.assertTrue(is_eval_source("data/raw/01-app/03-api-reference/03-file-conventions/layout.mdx"))
        self.assertTrue(is_eval_source("data/raw/02-pages/04-api-reference/01-components/link.mdx"))

    def test_other_pages_are_not_excluded(self):
        self.assertFalse(is_eval_source("data/raw/01-app/02-guides/streaming.mdx"))
        self.assertFalse(is_eval_source("data/raw/01-app/03-api-reference/04-functions/cookies.mdx"))

    def test_no_eval_page_appears_in_any_training_file(self):
        """Reads the real generated data. Skips if the data has not been built yet."""
        data_files = [
            ROOT / "data" / "pretrain" / "train.jsonl",
            ROOT / "data" / "pretrain" / "validation.jsonl",
            ROOT / "data" / "qa" / "train.jsonl",
            ROOT / "data" / "qa" / "validation.jsonl",
        ]
        if not all(p.exists() for p in data_files):
            self.skipTest("Training data not built yet")

        for path in data_files:
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                source = record.get("source", "")
                self.assertFalse(is_eval_source(source), f"eval page {source} found in {path.name}")


class CheckpointAndGenerationTests(unittest.TestCase):
    def test_checkpoint_round_trip_gives_identical_logits(self):
        torch.manual_seed(0)
        model = GPT(TINY).eval()
        idx = torch.randint(0, TINY.vocab_size, (1, 8))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            save_checkpoint(path, model, TINY, stage="pretrain", step=5, metrics={"val_loss": 1.0})
            loaded, meta = load_checkpoint(path)
        with torch.no_grad():
            a, _ = model(idx)
            b, _ = loaded(idx)
        self.assertTrue(torch.allclose(a, b))
        self.assertEqual(meta["stage"], "pretrain")
        self.assertEqual(meta["step"], 5)

    def test_greedy_generation_is_repeatable(self):
        torch.manual_seed(0)
        model = GPT(TINY).eval()
        prompt = torch.tensor([[2, 5, 9]])
        first = model.generate(prompt, max_new_tokens=6, temperature=0)
        second = model.generate(prompt, max_new_tokens=6, temperature=0)
        self.assertTrue(torch.equal(first, second))

    def test_evaluate_model_returns_one_record_per_question(self):
        torch.manual_seed(0)
        model = GPT(TINY).eval()
        tok = BPETokenizer.train(["Question: a. Answer: b. " * 20], vocab_size=300)
        questions = [
            {"id": 1, "question": "What is a layout?", "keywords": ["layout"], "reference": "A layout wraps pages."},
            {"id": 2, "question": "What is a page?", "keywords": ["page"], "reference": "A page renders a route."},
        ]
        records, summary = evaluate_model(model, tok, questions, max_new_tokens=4)
        self.assertEqual(len(records), 2)
        self.assertIn("accuracy", summary)
        self.assertEqual({r["id"] for r in records}, {1, 2})


if __name__ == "__main__":
    unittest.main()
