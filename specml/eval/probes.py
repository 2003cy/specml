"""Attention-pooling probes for downstream evaluation.

Mirrors the probes in downstream_tasks.ipynb: a learnable multi-head query
attends over the VALID token embeddings, then a small MLP head predicts the
target. The SpecML encoder stays frozen — only the pool + head are trained.

Training uses a train/val split with EARLY STOPPING on the validation metric;
the reported numbers (and the plots) come from the validation set at the best
step, so they don't depend on an arbitrary fixed step count.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    """One learnable (multi-head) query attends over the valid tokens."""

    def __init__(self, d, h=4):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q = nn.Parameter(torch.randn(h, self.dh) * 0.02)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)

    def forward(self, x, V):                       # x [B,T,D], V [B,T] bool
        B, T, _ = x.shape
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)   # [B,h,T,dh]
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        a = (self.q[None, :, None, :] * k).sum(-1) * self.dh ** -0.5  # [B,h,T]
        a = a.masked_fill(~V[:, None, :], -1e9).softmax(-1)
        return (a.unsqueeze(-1) * v).sum(2).reshape(B, -1)           # [B,D]


def _mlp_head(d, n_out):
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_out))


@torch.no_grad()
def _forward_batched(pool, head, tok, V, device, bs=512):
    """Forward the (frozen-input) probe over tok/V in batches → logits/preds [N, n_out], CPU."""
    pool.eval(); head.eval()
    out = []
    for i in range(0, len(tok), bs):
        out.append(head(pool(tok[i:i + bs].float().to(device), V[i:i + bs].to(device))).cpu())
    pool.train(); head.train()
    return torch.cat(out)


def train_regression(tok_tr, V_tr, y_tr, tok_val, V_val, y_val, d, device,
                     n_out=1, max_steps=4000, lr=1e-2, bs=256, seed=0,
                     eval_every=100, patience=15):
    """Train AttnPool+MLP with EARLY STOPPING on validation MSE.

    Returns the de-normalised VALIDATION predictions [N_val, n_out] at the best
    val step. tok_*: [N,T,D], V_*: [N,T] bool, y_tr: tensor [N(,n_out)],
    y_val: array [N(,n_out)].
    """
    torch.manual_seed(seed)
    pool = AttnPool(d).to(device)
    head = _mlp_head(d, n_out).to(device)
    opt = torch.optim.AdamW(list(pool.parameters()) + list(head.parameters()), lr=lr)

    y_tr = y_tr.reshape(len(y_tr), n_out).float()
    y_val = np.asarray(y_val, dtype=np.float32).reshape(len(y_val), n_out)
    ymean, ystd = y_tr.mean(0), y_tr.std(0).clamp(min=1e-8)
    y_tr_n = ((y_tr - ymean) / ystd).to(device)

    g = torch.Generator().manual_seed(seed)
    best_mse, best_pred, bad = float("inf"), None, 0
    pool.train(); head.train()
    for step in range(max_steps):
        b = torch.randint(len(tok_tr), (bs,), generator=g)
        pred = head(pool(tok_tr[b].float().to(device), V_tr[b].to(device)))
        loss = F.mse_loss(pred, y_tr_n[b])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % eval_every == 0:
            vp = (_forward_batched(pool, head, tok_val, V_val, device) * ystd + ymean).numpy()
            vmse = float(np.mean((vp - y_val) ** 2))
            if vmse < best_mse - 1e-7:
                best_mse, best_pred, bad = vmse, vp, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_pred is None:  # max_steps < eval_every guard
        best_pred = (_forward_batched(pool, head, tok_val, V_val, device) * ystd + ymean).numpy()
    return best_pred


def train_classifier(tok_tr, V_tr, y_tr, tok_val, V_val, y_val, d, device,
                     max_steps=3000, lr=1e-2, bs=256, seed=0,
                     eval_every=100, patience=15):
    """Binary classifier with EARLY STOPPING on validation BCE.

    Returns the VALIDATION logits (numpy) at the best val step.
    """
    torch.manual_seed(seed)
    pool = AttnPool(d).to(device)
    head = _mlp_head(d, 1).to(device)
    opt = torch.optim.AdamW(list(pool.parameters()) + list(head.parameters()), lr=lr)

    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(np.asarray(y_val, dtype=np.float32))
    pos = float(y_tr_t.mean()); pw = torch.tensor([(1 - pos) / max(pos, 1e-3)], device=device)

    g = torch.Generator().manual_seed(seed)
    best_bce, best_logit, bad = float("inf"), None, 0
    pool.train(); head.train()
    for step in range(max_steps):
        b = torch.randint(len(tok_tr), (bs,), generator=g)
        logit = head(pool(tok_tr[b].float().to(device), V_tr[b].to(device))).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, y_tr_t[b], pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % eval_every == 0:
            vl = _forward_batched(pool, head, tok_val, V_val, device).squeeze(-1)
            vbce = float(F.binary_cross_entropy_with_logits(vl, y_val_t, pos_weight=pw.cpu()))
            if vbce < best_bce - 1e-7:
                best_bce, best_logit, bad = vbce, vl.numpy(), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_logit is None:
        best_logit = _forward_batched(pool, head, tok_val, V_val, device).squeeze(-1).numpy()
    return best_logit


# ── metric helpers ───────────────────────────────────────────────────────────

def sigma_nmad(dz):
    """1.4826 · MAD — the robust scatter convention used in the notebook."""
    return float(1.4826 * np.median(np.abs(dz - np.median(dz))))


def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)
