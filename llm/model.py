"""A small decoder-only Transformer (GPT-style), randomly initialized.

Each block does:
    x = x + Attention(LayerNorm(x))     # tokens look at earlier tokens
    x = x + MLP(LayerNorm(x))           # each token is processed on its own

Tokens can only attend to earlier positions (the "causal" mask), so the model
learns to predict the next token from what came before it.
"""

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8000
    context_length: int = 512   # longest sequence the model can see
    n_layer: int = 8            # number of Transformer blocks
    n_head: int = 6             # attention heads per block
    n_embd: int = 288           # width of each token vector (must divide by n_head)
    dropout: float = 0.1
    tie_weights: bool = True    # reuse the input embedding as the output layer

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # One layer produces queries, keys and values together
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        # (B, T, C) -> (B, heads, T, head_dim)
        def heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)

        # is_causal=True applies the "no peeking at future tokens" mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale the output projections of each residual branch down with depth,
        # which keeps the activations from growing as the model gets deeper.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. targets: (B, T) next-token ids, or None.

        Returns (logits, loss). loss is None when no targets are given.
        Positions with target -100 are ignored, which is how prompts are masked.
        """
        B, T = idx.shape
        if T > self.cfg.context_length:
            raise ValueError(f"sequence length {T} exceeds context_length {self.cfg.context_length}")

        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_parameters(self):
        # Shared (tied) weights are counted once
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """Generate new tokens one at a time, feeding each back in.

        temperature=0 means greedy decoding (always the most likely token),
        which is what the evaluation uses so results are repeatable.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = float("-inf")
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return idx


def config_to_dict(cfg):
    return asdict(cfg)
