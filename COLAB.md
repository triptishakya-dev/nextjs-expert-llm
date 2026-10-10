# Training the GPT on Google Colab (T4 GPU), two stages

Stage 1 pretrains on raw Next.js docs. Stage 2 continues from that checkpoint on the Q&A data.
Both stages save checkpoints in their own folder.

## Files to upload

`data/` and `models/` are gitignored, so upload these to Colab (or to Google Drive and copy them in). They are the exact files validated locally:

- `data/pretrain/train.jsonl`, `data/pretrain/validation.jsonl`
- `data/qa/train.jsonl`, `data/qa/validation.jsonl`
- `models/gpt/tokenizer.json`

Everything else comes from the repository.

## Cell 1: GPU and Drive

```bash
nvidia-smi
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

Runtime -> Change runtime type -> **T4 GPU** before running.

## Cell 2: code and data

Clone the repository (or upload it as a zip), then put the data in place. Replace `<your-repo-url>`:

```bash
git clone <your-repo-url> /content/next-js-expert-llm
cd /content/next-js-expert-llm
mkdir -p data/pretrain data/qa models/gpt
```

Then upload the files above into those folders.

## Cell 3: unit tests (must pass before training)

```bash
cd /content/next-js-expert-llm && python -m unittest discover -s tests/unit -t .
```

## Cell 4: smoke test for both stages (about 2 minutes)

```bash
cd /content/next-js-expert-llm && python scripts/train_gpt.py --stage pretrain --max-steps 20 --eval-every 10 --skip-eval --out-dir /content/smoke/pretrain
```

```bash
cd /content/next-js-expert-llm && python scripts/train_gpt.py --stage instruct --init-from /content/smoke/pretrain/best.pt --max-steps 20 --eval-every 10 --skip-eval --out-dir /content/smoke/instruct
```

## Cell 5: stage 1, pretraining (2,000 steps)

```bash
cd /content/next-js-expert-llm && python scripts/train_gpt.py --stage pretrain --out-dir /content/drive/MyDrive/nextjs-gpt/pretrain
```

Checkpoints: `best.pt` (lowest validation loss) and `last.pt`. Logs: `metrics.jsonl`. Summary: `summary.json`. Keyword evaluation: `keyword_eval.json`.

## Cell 6: stage 2, instruction tuning (1,500 steps)

```bash
cd /content/next-js-expert-llm && python scripts/train_gpt.py --stage instruct --init-from /content/drive/MyDrive/nextjs-gpt/pretrain/best.pt --out-dir /content/drive/MyDrive/nextjs-gpt/instruct
```

## Cell 7: evaluation of both checkpoints (already run at the end of each stage; repeat to re-check)

```bash
cd /content/next-js-expert-llm && python scripts/evaluate_gpt.py --checkpoint /content/drive/MyDrive/nextjs-gpt/pretrain/best.pt --name pretrain --output /content/drive/MyDrive/nextjs-gpt/pretrain/keyword_eval.json
```

```bash
cd /content/next-js-expert-llm && python scripts/evaluate_gpt.py --checkpoint /content/drive/MyDrive/nextjs-gpt/instruct/best.pt --name instruct --output /content/drive/MyDrive/nextjs-gpt/instruct/keyword_eval.json
```

## What to watch

- **Train loss** should fall steadily in both stages.
- **Validation loss** should fall at first, then level off. If it rises while train loss keeps falling, the model is overfitting. `best.pt` is already the lowest-validation checkpoint, so you can stop early and use it.
- Stage 1 validation loss is a per-token loss on raw docs. Stage 2 validation loss is on answer tokens only, so the two numbers are not directly comparable.
- The keyword evaluation is the comparison to the Qwen-era numbers (accuracy, keyword recall, relevance F1). Compare against the random-init baseline in `tests/results/gpt-random-init.json`.
- Timing: not measured on a T4 yet. Measure it on the smoke run (Cell 4) and extrapolate.
