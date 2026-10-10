"""Scoring and answer generation for the GPT evaluation.

The scoring functions are the same as scripts/evaluate.py (the Qwen-era eval),
so results from the two models can be compared directly:

- accuracy: share of questions whose answer contains every expected keyword
- keyword_recall: average share of expected keywords found per answer
- relevance_f1: unigram F1 between the answer and the reference answer
"""

import re

import torch

from llm.bpe import BOS_ID, EOS_ID
from llm.dataset import format_prompt

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "you",
    "can", "use", "used", "which", "its", "into", "your", "they", "when",
    "such", "what", "does", "not", "all", "any", "has", "have", "will",
}


def tokenize(text):
    words = re.findall(r"[a-z0-9@\[\]]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def relevance_f1(answer, reference):
    answer_tokens = set(tokenize(answer))
    reference_tokens = set(tokenize(reference))

    if not answer_tokens or not reference_tokens:
        return 0.0

    overlap = len(answer_tokens & reference_tokens)
    if overlap == 0:
        return 0.0

    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(answer, question):
    text = answer.lower()
    keywords = [k.lower() for k in question["keywords"]]
    found = [k for k in keywords if k in text]

    return {
        "correct": len(found) == len(keywords),
        "keyword_recall": len(found) / len(keywords),
        "relevance_f1": relevance_f1(answer, question["reference"]),
    }


def summarize(scored):
    total = len(scored)
    return {
        "accuracy": sum(s["correct"] for s in scored) / total,
        "keyword_recall": sum(s["keyword_recall"] for s in scored) / total,
        "relevance_f1": sum(s["relevance_f1"] for s in scored) / total,
    }


@torch.no_grad()
def generate_answer(model, tokenizer, question, max_new_tokens, device="cpu"):
    """Greedy answer to one question, using the same prompt format as training."""
    prompt = [BOS_ID] + tokenizer.encode(format_prompt(question))
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0, eos_id=EOS_ID)
    return tokenizer.decode(out[0, len(prompt):].tolist()).strip()


def evaluate_model(model, tokenizer, questions, max_new_tokens=160, device="cpu"):
    """Return (per-question records, summary metrics)."""
    records = []
    scored = []
    for item in questions:
        answer = generate_answer(model, tokenizer, item["question"], max_new_tokens, device)
        scores = score_answer(answer, item)
        scored.append(scores)
        records.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            **scores,
        })
    return records, summarize(scored)
