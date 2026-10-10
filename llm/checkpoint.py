"""Save and load GPT checkpoints.

A checkpoint is one file holding the weights, the model config, and some
metadata (which stage made it, which step, and the validation loss).
Loading rebuilds the model from the config, so no code has to be remembered.
"""

from pathlib import Path

import torch

from llm.model import GPT, GPTConfig


def save_checkpoint(path, model, cfg, stage, step, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": dict(cfg.__dict__),
        "stage": stage,
        "step": step,
        "metrics": metrics or {},
    }, path)


def load_checkpoint(path, device="cpu"):
    """Return (model in eval mode, metadata dict) for a checkpoint file."""
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**payload["config"])
    model = GPT(cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k not in ("model_state", "config")}
    meta["config"] = payload["config"]
    return model, meta
