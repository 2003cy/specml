import numpy as np 
import torch

from SpecML import SpecML
from Tokeniser import patch_size, step_size, P, V, X
import time
import optuna
import torch.nn as nn


start_time = time.time()
#------------------------------------------TRAINING PARAMETERS----------------------------------------------------#

N_STEPS_PER_RESTART = 20000  # gradient steps
BATCH_SIZE = 128  # spectra per batch
LR = 5e-4  # AdamW learning rate
WEIGHT_DECAY = 0.01  # AdamW weight decay
BETAS = (0.9, 0.95)  # AdamW β₁, β₂
GRAD_CLIP = 1.0  # gradient clip max norm
SCHED_ETA_MIN = 1e-6  # minimum LR after annealing
NUM_RESTARTS = 4  # number of annealing cycles
N_STEPS = N_STEPS_PER_RESTART * NUM_RESTARTS

#-------------------------------------------------MASKING-----------------------------------------------------------#

CHUNK_WIDTH = int(np.floor(2.5 * patch_size / step_size)) # chunk width in tokens
MASK_RATIO = 0.75  # fraction of max_chunks to mask per spectrum

def apply_chunk_mask_batch(
    y_b: torch.Tensor,
    v_b: torch.Tensor,
    mask_ratio: float = MASK_RATIO,
    chunk_width: int = CHUNK_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask one batch entirely on device"""
    B, T, _ = y_b.shape
    device = y_b.device
    M = torch.zeros(B, T, dtype=torch.bool, device=device)

    valid_lens = v_b.sum(dim=1).int()
    n_chunks = valid_lens // chunk_width
    k = ((((n_chunks + 1) // 2) * mask_ratio).int()).clamp(min=1)
    section = valid_lens // k.clamp(min=1)
    usable = (valid_lens >= chunk_width) & (section >= chunk_width)

    offsets = torch.arange(chunk_width, device=device)
    k_max = int(k[usable].max()) if usable.any() else 0

    for j in range(k_max):
        active = torch.where(usable & (k > j))[0]
        sec = section[active]
        u = torch.rand(len(active), device=device)
        starts = (j * sec + u * (sec - chunk_width + 1)).int()
        M[active[:, None], starts[:, None] + offsets] = True

    x_b = y_b.clone()
    x_b[M] = 0.0
    return x_b, M


#-------------------------------------------------LOSS-----------------------------------------------------------#

def mse_loss(Y, Yhat, M):
    err = ((Y - Yhat) ** 2).sum(dim=-1)  # [B, T]  squared L2 norm over patch dim
    return err[M].mean()  # mean over masked positions only


#-------------------------------------------------TRAINING-----------------------------------------------------------#

def train_trial(trial, device):
    # Tokenisation hyperparameters
    patch_s, overlap = [int(x) for x in trial.suggest_categorical('stride', [
    f"{p},{max(1, int(p * (op / 100)))}"
    for p in range(4, 21, 2)
    for op in range(10, 60, 10)
    ]).split(',')]
    step_s = patch_s - overlap

    # Model hyperparameters
    embedding_dim      = trial.suggest_categorical('embedding_dim',       [64, 128, 256])
    transformer_layers = trial.suggest_categorical('transformer_layers',  [2, 4, 6])
    batch_s            = trial.suggest_categorical('batch_size',          [64, 128, 256])
    lr                 = trial.suggest_categorical('learning_rate',       [1e-4, 5e-4, 1e-3])
    trial_steps        = trial.suggest_categorical('steps',               [20000, 40000, 60000])

    # ------ Retokenise for this trial's patch_size / overlap ------
    from numpy.lib.stride_tricks import sliding_window_view
    from astropy.table import Table
    from Tokeniser import data, valid_spectrum, valid_w, w, f, f_norm, dq

    x_t  = sliding_window_view(f_norm, patch_s, axis=1)[:, ::step_s]
    X    = np.concatenate([np.nanmean(x_t, axis=2, keepdims=True),
                           np.nanstd(x_t,  axis=2, keepdims=True),
                           x_t], axis=2)                              # (B, T, patch_s+2)
    V    = np.array(sliding_window_view(dq, patch_s, axis=1)[:, ::step_s].all(axis=2))
    X[~V] = 0.0

    # Positional encoding
    w_patches = sliding_window_view(w, patch_s)[::step_s].mean(axis=1)
    omegas    = 10000 ** (-2 * np.arange(embedding_dim // 2) / embedding_dim)
    product   = np.outer(w_patches * 1e4, omegas)
    P         = np.empty((X.shape[1], embedding_dim))
    P[:, 0::2] = np.sin(product)
    P[:, 1::2] = np.cos(product)

    Y_dev = torch.from_numpy(X).float().to(device)
    V_dev = torch.from_numpy(V).bool().to(device)
    P_dev = torch.from_numpy(P).float().to(device)
    N     = Y_dev.shape[0]

    # ------ Build model for this trial ------
    from SpecML import SpectralBlock

    class SpecMLTrial(nn.Module):
        def __init__(self, patch_dim, d=embedding_dim, h=4,
                     n_layers=transformer_layers, ff=4*embedding_dim):
            super().__init__()
            self.embed  = nn.Linear(patch_dim, d)
            nn.init.trunc_normal_(self.embed.weight, std=1/d, a=-3/d, b=3/d)
            self.blocks = nn.ModuleList([SpectralBlock(d, h, ff) for _ in range(n_layers)])
            self.norm   = nn.LayerNorm(d)
            self.head   = nn.Linear(d, patch_dim)

        def _encode(self, X, V, P):
            x = self.embed(X) + P
            for blk in self.blocks:
                x = blk(x, V)
            return self.norm(x)

        def forward(self, X, V, P):
            return self.head(self._encode(X, V, P))

        def encode(self, X, V, P):
            x    = self._encode(X, V, P)
            mask = V.unsqueeze(-1).to(x.dtype)
            return (x * mask).sum(dim=1) / mask.sum(dim=1)

    model     = SpecMLTrial(patch_dim=patch_s + 2).to(device)
    opt       = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY, betas=BETAS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=max(trial_steps // 2, 1), eta_min=SCHED_ETA_MIN)
    chunk_w   = int(np.floor(2.5 * patch_s / step_s))

    # ------ Training loop ------
    rng       = torch.Generator(device=device)
    rng.manual_seed(0)
    step      = 0
    best_loss = float('inf')

    while step < trial_steps:
        perm = torch.randperm(N, device=device, generator=rng)
        for start in range(0, N, batch_s):
            idx          = perm[start : start + batch_s]
            y_b, v_b     = Y_dev[idx], V_dev[idx]
            x_b, m_b     = apply_chunk_mask_batch(y_b, v_b,
                               mask_ratio=MASK_RATIO, chunk_width=chunk_w)
            loss         = mse_loss(y_b, model(x_b, v_b, P_dev), m_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            scheduler.step()
            best_loss = min(best_loss, loss.item())
            step += 1
            if step >= trial_steps:
                break

        trial.report(best_loss, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_loss


if __name__ == '__main__':
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available():
        device = 'mps'

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2000),
    )
    study.optimize(lambda trial: train_trial(trial, device), n_trials=10)

    print("\n" + "=" * 60)
    print("OPTUNA COMPLETE")
    print("=" * 60)
    print(f"Best loss:  {study.best_value:.6f}")
    print(f"Best trial: #{study.best_trial.number}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    import json
    with open('best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)

    end_time = time.time() - start_time
    print(f"\nTotal time: {end_time:.2f}s")

