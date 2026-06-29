"""Transformer building blocks for SpecML (attention + block)."""

import torch.nn as nn
import torch.nn.functional as F


class SpectralAttention(nn.Module):
    def __init__(self, d, h):  # d and h defined in SpecML
        super().__init__()
        assert d % h == 0  # Error if not True
        self.h, self.dh = h, d // h  # number of heads / per-head dim
        self.qkv = nn.Linear(d, 3 * d)  # learnt QKV matrix (one tensor, split into Q,K,V)
        self.out = nn.Linear(d, d)  # learnt output projection
        # Local positional embedding: per-head, per-channel 3-tap filter.
        self.local = nn.Conv1d(self.dh, self.dh, 3, padding=1, groups=self.dh, bias=False)

    def forward(self, x, validity):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)  # [B, H, T, dh]
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)

        # Depthwise conv on V, applied independently per head.
        v_flat = v.reshape(B * self.h, T, self.dh).transpose(1, 2)  # [B*H, dh, T]
        local = self.local(v_flat).transpose(1, 2).view(B, self.h, T, self.dh)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=validity[:, None, None, :]
        )  # [B, H, T, dh]
        y = y + local
        y = y * validity[:, None, :, None].to(x.dtype)  # zero invalid queries

        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out(y)


class SpectralBlock(nn.Module):
    def __init__(self, d, h, ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = SpectralAttention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, validity):
        x = x + self.drop(self.attn(self.ln1(x), validity))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x
