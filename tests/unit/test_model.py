import json
import tempfile
import unittest
from pathlib import Path

import torch

from llm.dataset import IGNORE_INDEX, QADataset, collate, encode_record
from llm.bpe import BPETokenizer
from llm.model import GPT, GPTConfig

SMALL = GPTConfig(vocab_size=500, context_length=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


class GPTModelTests(unittest.TestCase):
    def test_default_config_is_about_ten_million_parameters(self):
        model = GPT(GPTConfig())
        n = model.num_parameters()
        self.assertGreater(n, 9_000_000)
        self.assertLess(n, 11_000_000)

    def test_output_shape(self):
        model = GPT(SMALL).eval()
        idx = torch.randint(0, SMALL.vocab_size, (3, 10))
        logits, loss = model(idx)
        self.assertEqual(logits.shape, (3, 10, SMALL.vocab_size))
        self.assertIsNone(loss)

    def test_loss_is_finite_at_start(self):
        # A random model should be close to uniform: loss near ln(vocab)
        model = GPT(SMALL).eval()
        idx = torch.randint(0, SMALL.vocab_size, (4, 16))
        _, loss = model(idx, idx)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(float(SMALL.vocab_size))).item(), delta=0.5)

    def test_causal_mask_future_tokens_do_not_change_past_logits(self):
        torch.manual_seed(0)
        model = GPT(SMALL).eval()
        a = torch.randint(0, SMALL.vocab_size, (1, 12))
        b = a.clone()
        b[0, 8:] = (b[0, 8:] + 1) % SMALL.vocab_size  # change only positions 8 onward
        with torch.no_grad():
            logits_a, _ = model(a)
            logits_b, _ = model(b)
        self.assertTrue(torch.allclose(logits_a[0, :8], logits_b[0, :8], atol=1e-5))

    def test_ignored_targets_do_not_contribute_to_loss(self):
        model = GPT(SMALL).eval()
        idx = torch.randint(0, SMALL.vocab_size, (1, 10))
        targets = idx.clone()
        targets[0, :5] = IGNORE_INDEX
        _, loss_masked = model(idx, targets)
        targets_only_tail = idx.clone()
        targets_only_tail[0, :5] = IGNORE_INDEX
        _, loss_same = model(idx, targets_only_tail)
        self.assertAlmostEqual(loss_masked.item(), loss_same.item(), places=6)

    def test_sequence_longer_than_context_is_rejected(self):
        model = GPT(SMALL)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, SMALL.context_length + 1), dtype=torch.long))

    def test_one_training_step_lowers_loss_on_same_batch(self):
        torch.manual_seed(0)
        model = GPT(SMALL).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        idx = torch.randint(0, SMALL.vocab_size, (4, 16))
        _, first = model(idx, idx)
        opt.zero_grad()
        first.backward()
        opt.step()
        _, second = model(idx, idx)
        self.assertLess(second.item(), first.item())

    def test_generate_returns_requested_length(self):
        model = GPT(SMALL).eval()
        out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        self.assertEqual(out.shape, (1, 8))

    def test_invalid_head_split_is_rejected(self):
        with self.assertRaises(ValueError):
            GPTConfig(n_embd=10, n_head=3)


class QADatasetTests(unittest.TestCase):
    def test_loss_is_only_on_answer_tokens(self):
        tok = BPETokenizer.train(["Question: what? Answer: a page. " * 30], vocab_size=300)
        record = {"question": "What is a layout?", "answer": "A shared UI wrapper.", "source": "x.mdx"}
        prompt_ids, answer_ids = encode_record(tok, record)
        self.assertEqual(prompt_ids[0], 2)  # <bos>
        self.assertEqual(answer_ids[-1], 3)  # <eos>

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qa.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = QADataset(path, tok, context_length=256)

        x, y = dataset[0]
        # Targets are shifted by one, so the first answer token is predicted at index len(prompt) - 1
        first_answer = len(prompt_ids) - 1
        self.assertTrue(all(t == IGNORE_INDEX for t in y[:first_answer]))
        self.assertEqual(y[first_answer:], answer_ids)
        self.assertEqual(len(x), len(y))

    def test_collate_pads_inputs_and_masks_padded_targets(self):
        batch = [([1, 2, 3], [5, 6, 7]), ([1, 2], [8, 9])]
        x, y = collate(batch)
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x[1].tolist(), [1, 2, 0])
        self.assertEqual(y[1].tolist(), [8, 9, IGNORE_INDEX])


if __name__ == "__main__":
    unittest.main()
