import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.table import Table

from SpecML import D_emb, SpecML
from Tokeniser import patch_size, P, V, X, valid_spectrum

# ---- Load catalog and align to valid spectra ---------------------------------
catalog = Table.read('dja_msaexp_emission_lines_v4.5.csv.gz', format='ascii')
catalog = catalog[catalog['grating'] == 'PRISM']
catalog = catalog[valid_spectrum]

mask_g3 = np.array(
    (np.array(catalog['grade'].filled(0)) == 3) & (np.array(catalog['z_best']) >= 0)
)
z_g3 = torch.from_numpy(np.array(catalog['z_best'][mask_g3], dtype=np.float32))

# ---- Encode grade-3 spectra with frozen model --------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'

model = SpecML(patch_dim=patch_size + 2).to(device)
model.load_state_dict(
    torch.load('SpecML.pt', map_location=device, weights_only=True)
)
model.eval()

with torch.no_grad():
    emb = model.encode(
        torch.from_numpy(X[mask_g3]).float().to(device),
        torch.from_numpy(V[mask_g3]).bool().to(device),
        torch.from_numpy(P).float().to(device),
    ).cpu()

# ---- 50/50 train/test split --------------------------------------------------
n = len(z_g3)
idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
split = n // 2

emb_train, emb_test = emb[idx[:split]], emb[idx[split:]]
z_train, z_test = z_g3[idx[:split]], z_g3[idx[split:]]

# ---- Normalize targets to stabilise optimisation ----------------------------
z_mean, z_std = z_train.mean(), z_train.std()
z_train_n = (z_train - z_mean) / z_std

# ---- Linear head, encoder frozen --------------------------------------------
head = nn.Sequential(nn.LayerNorm(D_emb), nn.Linear(D_emb, 1))
opt = torch.optim.AdamW(head.parameters(), lr=1e-2) #changed from 1e-3

head.train()
for step in range(2000):
    batch = torch.randint(len(emb_train), (256,)) 
    loss = F.mse_loss(head(emb_train[batch]).squeeze(-1), z_train_n[batch])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f'step {step:4d}  loss {loss.item():.4f}')

# ---- Evaluate ----------------------------------------------------------------
head.eval()
with torch.no_grad():
    z_pred = head(emb_test).squeeze(-1) * z_std + z_mean

dz = (z_pred - z_test).abs() / (1 + z_test)
print(f'grade-3 test set:  N={len(z_test)}')
print(f'MAE                {(z_pred - z_test).abs().mean().item():.4f}')
print(f'median |Δz|/(1+z)  {dz.median().item():.4f}')

# ---- Plot --------------------------------------------------------------------

z_true_np = z_test.numpy()
z_pred_np = z_pred.numpy()
dz_np = dz.numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
lim = (0, max(z_true_np.max(), z_pred_np.max()) * 1.05)
ax.scatter(
    z_true_np, z_pred_np, c=dz_np, cmap='plasma', s=8, alpha=0.7, vmin=0, vmax=0.1
)
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('z_true')
ax.set_ylabel('z_pred')
ax.set_title('True vs predicted redshift')
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 0.1))
fig.colorbar(sm, ax=ax, label='|Δz|/(1+z)')

ax = axes[1]
ax.hist(dz_np, bins=40, range=(0, 0.3), color='steelblue', edgecolor='none')
ax.axvline(
    float(dz.median()), color='red', lw=1.2, label=f'median={float(dz.median()):.4f}'
)
ax.set_xlabel('|Δz| / (1+z)')
ax.set_ylabel('count')
ax.set_title('Redshift error distribution')
ax.legend()

plt.tight_layout()
plt.savefig('downstream_linear.png', dpi=150)
plt.show()
print('Saved downstream_linear.png')












