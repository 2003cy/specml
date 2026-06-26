import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.table import Table

from SpecML import load_specml
from Tokeniser import f, dq, w, valid_spectrum, valid_spectra, tokenize, normalise_flux

# ---- Load model (params come from the checkpoint itself) --------------------
model_file = 'SpecML.pt'

device = ('cuda' if torch.cuda.is_available() else
          'mps'  if torch.backends.mps.is_available() else 'cpu')

model, cfg = load_specml(model_file, device=device)

# ---- Tokenise with the SAME preprocessing + patch params used in training ----
f_norm = normalise_flux(f)   # arcsinh stretch + z-score (matches Tokeniser/Training)
X, V, P = tokenize(f_norm, dq, w, cfg['patch_size'], cfg['overlap'], cfg['D_emb'])

# ---- Load catalog and align to valid spectra --------------------------------
catalog = Table.read('dja_msaexp_emission_lines_v4.5.csv.gz', format='ascii')
catalog = catalog[catalog['grating'] == 'PRISM']
catalog = catalog[valid_spectrum][valid_spectra]

# ---- Target: Hα+NII line (line_ha_nii), require S/N >= 3 --------------------
ha     = np.array(catalog['line_ha_nii'],     dtype=np.float32)
ha_err = np.array(catalog['line_ha_nii_err'], dtype=np.float32)
mask_ew = np.isfinite(ha) & np.isfinite(ha_err) & (ha_err > 0) & (ha >= 3 * ha_err)
y_ha = torch.from_numpy(ha[mask_ew])

# ---- Encode selected spectra ------------------------------------------------
with torch.no_grad():
    emb = model.encode(
        torch.from_numpy(X[mask_ew]).float().to(device),
        torch.from_numpy(V[mask_ew]).bool().to(device),
        torch.from_numpy(P).float().to(device),
    ).cpu()

# ---- 50/50 train/test split --------------------------------------------------
n = len(y_ha)
idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
split = n // 2

emb_train, emb_test = emb[idx[:split]], emb[idx[split:]]
y_train, y_test = y_ha[idx[:split]], y_ha[idx[split:]]

# ---- Normalize targets to stabilise optimisation ----------------------------
y_mean, y_std = y_train.mean(), y_train.std()
y_train_n = (y_train - y_mean) / y_std

# ---- Linear head, encoder frozen --------------------------------------------
head = nn.Sequential(nn.LayerNorm(cfg['D_emb']), nn.Linear(cfg['D_emb'], 1))
opt = torch.optim.AdamW(head.parameters(), lr=1e-2)

head.train()
for step in range(2000):
    batch = torch.randint(len(emb_train), (256,))
    loss = F.mse_loss(head(emb_train[batch]).squeeze(-1), y_train_n[batch])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f'step {step:4d}  loss {loss.item():.4f}')

# ---- Evaluate ----------------------------------------------------------------
head.eval()
with torch.no_grad():
    y_pred = head(emb_test).squeeze(-1) * y_std + y_mean

print(f'test set:  N={len(y_test)}')
print(f'MAE                {(y_pred - y_test).abs().mean().item():.4f}')

# ---- Plot --------------------------------------------------------------------
y_true_np = y_test.numpy()
y_pred_np = y_pred.numpy()
resid_np = np.abs(y_pred_np - y_true_np)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
lim = (min(y_true_np.min(), y_pred_np.min()),
       max(y_true_np.max(), y_pred_np.max()) * 1.05)
ax.scatter(y_true_np, y_pred_np, c=resid_np, cmap='plasma', s=8, alpha=0.7)
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('line_ha_nii true')
ax.set_ylabel('line_ha_nii pred')
ax.set_title('True vs predicted Hα+NII line')
fig.colorbar(plt.cm.ScalarMappable(cmap='plasma'), ax=ax, label='|pred - true|')

ax = axes[1]
ax.hist(resid_np, bins=40, color='steelblue', edgecolor='none')
ax.axvline(
    float(np.median(resid_np)), color='red', lw=1.2,
    label=f'median={float(np.median(resid_np)):.4f}'
)
ax.set_xlabel('|pred - true|')
ax.set_ylabel('count')
ax.set_title('Hα+NII error distribution')
ax.legend()

plt.tight_layout()
plt.savefig('downstream_linear_Ha.png', dpi=150)
plt.show()
print('Saved downstream_linear_Ha.png')
