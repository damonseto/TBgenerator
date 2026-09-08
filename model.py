"""ClimbGPT model and constrained decoder.

Moved verbatim out of 02.ipynb in the tensionboardproject repo so a server can
import it. The only change is that generate() takes token_to_hold as an
argument instead of reading it from notebook globals.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import BOS, EOS, role_to_token, decode


class AttentionHead(nn.Module):
    """One attention head. Same as before, just narrower."""
    def __init__(self, d_model, d_head):
        super().__init__()
        self.query = nn.Linear(d_model, d_head)
        self.key   = nn.Linear(d_model, d_head)
        self.value = nn.Linear(d_model, d_head)
        self.d_head = d_head

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

        T = x.shape[1]
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        return weights @ v

class MultiHeadAttention(nn.Module):
    """Several heads in parallel, glued back together.

    Each head gets d_model // n_heads dimensions, so the total width is
    unchanged — you're splitting the same capacity across independent views
    rather than adding any.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        d_head = d_model // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, d_head) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(d_model, d_model)   # let the heads mix

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class Block(nn.Module):
    """One transformer layer: attention, then per-position processing.

    Attention moves information between positions. The MLP then processes each
    position on its own. Both are wrapped in residual connections, so each
    sublayer adds to the running representation rather than replacing it.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ClimbGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=42):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)   # token number -> vector
        self.pos_emb   = nn.Embedding(max_len + 1, d_model)  # position -> vector
        self.cond      = nn.Linear(2, d_model)               # (angle, grade) -> vector

        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)           # -> score per token

    def forward(self, tokens, cond):
        # tokens: (B, T) integers.  cond: (B, 2) floats, angle and grade in 0-1.
        B, T = tokens.shape

        x = self.token_emb(tokens)                           # (B, T, d_model)

        # conditioning goes in as an extra position at the front
        c = self.cond(cond).unsqueeze(1)                      # (B, 1, d_model)
        x = torch.cat([c, x], dim=1)                          # (B, T+1, d_model)

        pos = torch.arange(T + 1, device=tokens.device)
        x = x + self.pos_emb(pos)                             # add position info

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)                                   # (B, T+1, vocab_size)


@torch.no_grad()
def generate(model, token_to_hold, angle, grade, temperature=0.8, max_holds=20):
    model.eval()
    cond = torch.tensor([[angle / 70.0, grade / 40.0]], dtype=torch.float)
    tokens = torch.tensor([[BOS]], dtype=torch.long)

    used, n_start, n_finish = set(), 0, 0

    for _ in range(2 * max_holds):
        logits = model(tokens, cond)[0, -1].clone()

        expecting_role = (tokens.shape[1] % 2 == 0)

        if expecting_role:
            allowed = set(role_to_token.values())
            if n_start >= 2:
                allowed.discard(role_to_token[5])
            if n_finish >= 1:
                allowed.discard(role_to_token[7])
        else:
            allowed = set(token_to_hold.keys()) - used
            if len(used) >= 4 and n_start >= 1 and n_finish >= 1:
                allowed.add(EOS)

        mask = torch.full_like(logits, float('-inf'))
        for t in allowed:
            mask[t] = 0.0

        probs = F.softmax((logits + mask) / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).item()

        if nxt == EOS:
            break

        if nxt in token_to_hold:
            used.add(nxt)
        elif nxt == role_to_token[5]:
            n_start += 1
        elif nxt == role_to_token[7]:
            n_finish += 1

        tokens = torch.cat([tokens, torch.tensor([[nxt]])], dim=1)

    return decode(tokens[0].tolist() + [EOS], token_to_hold)
