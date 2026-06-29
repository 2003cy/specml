"""Downstream evaluation for a SpecML checkpoint (frozen encoder + probes).

Mirrors downstream_tasks.ipynb: trains attention-pooling probes on the FROZEN
SpecML token embeddings and reports the headline metrics.

  * Redshift:  grade-3 (z>=0) galaxies, 50/50 split, AttnPool+MLP regressor.
               Reports MAE, median |Δz|/(1+z), σ_NMAD, catastrophic (>0.15), R².
  * BPT:       S/N>=3 on Hβ, [OIII]5007, [SII], Hα+NII at z>=1. A regressor
               predicts the two BPT axes (MAE); a classifier (trained on the
               low-z anchor) separates SF vs AGN (anchor-val accuracy).

Usage:
    python eval/run_eval.py --ckpt output/<run>/version_0/checkpoints/last.ckpt
    python eval/run_eval.py --ckpt <ckpt> --task redshift --seeds 0 1 2
"""

import argparse
import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from astropy.table import Table

from data.dataset import SpecMLDataset
from model.specml import load_specml
from eval.probes import train_regression, train_classifier, sigma_nmad, r2_score

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CATALOG = os.path.join(_REPO_ROOT, "dja_msaexp_emission_lines_v4.5.csv.gz")


# ── frozen token embeddings ──────────────────────────────────────────────────

@torch.no_grad()
def encode_tokens(model, flux, wave, valid, idx, device, bs=256):
    """Encode the spectra at rows `idx` → (tokens [N,T,D] f16, V [N,T] bool), CPU."""
    toks, masks = [], []
    wave_t = torch.from_numpy(wave).float().to(device)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        fb = torch.from_numpy(flux[j]).float().to(device)
        wb = wave_t.unsqueeze(0).expand(len(j), -1)
        vb = torch.from_numpy(valid[j]).bool().to(device)
        out = model.embed(fb, wb, vb)
        toks.append(out["tokens"].half().cpu())
        masks.append(out["token_valid_mask"].cpu())
    return torch.cat(toks), torch.cat(masks)


# ── tasks ────────────────────────────────────────────────────────────────────

def eval_redshift(model, ds, cat, device, seeds, outdir=None):
    z_best = np.array(cat["z_best"], dtype=np.float32)
    grade = np.array(cat["grade"].filled(0) if hasattr(cat["grade"], "filled") else cat["grade"]).astype(int)
    sel = np.where((grade == 3) & (z_best >= 0) & ds.valid.any(axis=1))[0]
    print(f"[redshift] grade-3 galaxies with z>=0: N={len(sel):,}")

    tok, V = encode_tokens(model, ds.flux, ds.wavelength, ds.valid, sel, device)
    z = torch.from_numpy(z_best[sel])
    d = model.hparams.embed_dim

    # Seeded 50/50 train/val split; the probe early-stops on val and we report
    # the val numbers (the val set is the held-out probe set — no separate test).
    n = len(sel)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    tr, va = idx[:n // 2], idx[n // 2:]
    z_va = z[va].numpy()

    runs = []
    for s in seeds:
        z_pred = train_regression(tok[tr], V[tr], z[tr], tok[va], V[va], z_va, d, device,
                                  n_out=1, seed=s).squeeze(-1)
        dz = (z_pred - z_va) / (1 + z_va)
        m = dict(seed=int(s),
                 MAE=float(np.abs(z_pred - z_va).mean()),
                 median_dz=float(np.median(np.abs(dz))),
                 sigma_nmad=sigma_nmad(dz),
                 catastrophic_0p15=float(np.mean(np.abs(dz) > 0.15)),
                 r2=r2_score(z_va, z_pred))
        print(f"  seed {s}: σ_NMAD={m['sigma_nmad']:.4f}  median|Δz|/(1+z)={m['median_dz']:.4f}  "
              f"cat>0.15={m['catastrophic_0p15']:.2%}  R²={m['r2']:.3f}")
        runs.append(m)
        last_pred = z_pred

    if outdir is not None:
        _plot_redshift(z_va, last_pred, runs[-1], os.path.join(outdir, "eval_redshift.png"))
    agg = {k: float(np.mean([r[k] for r in runs])) for k in
           ("MAE", "median_dz", "sigma_nmad", "catastrophic_0p15", "r2")}
    return {"N": int(n), "per_seed": runs, "mean": agg}


def eval_bpt(model, ds, cat, device, seed, outdir, min_snr=3.0, z_min=1.0):
    def col(name, fill=np.nan):
        c = cat[name]; a = np.array(c.filled(fill) if hasattr(c, "filled") else c, dtype=np.float32)
        a[~np.isfinite(a)] = fill; return a

    def err(name, fill=np.inf):
        c = cat[name]; a = np.array(c.filled(fill) if hasattr(c, "filled") else c, dtype=np.float32)
        a[~np.isfinite(a) | (a <= 0)] = fill; return a

    hb_f, hb_e = col("line_hb"), err("line_hb_err")
    _OIII_5007 = 2.98 / (1.0 + 2.98)                       # de-blend [OIII]5007 from 4959+5007
    oiii_f, oiii_e = col("line_oiii") * _OIII_5007, err("line_oiii_err") * _OIII_5007
    sii_f, sii_e = col("line_sii"), err("line_sii_err")
    han_f, han_e = col("line_ha_nii"), err("line_ha_nii_err")
    z_all = col("z_best", fill=-1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        x_true = np.log10(sii_f / han_f).astype(np.float32)     # log([SII]/(Hα+NII))
        y_true = np.log10(oiii_f / hb_f).astype(np.float32)     # log([OIII]/Hβ)
    snr_ok = ((hb_f / hb_e >= min_snr) & (oiii_f / oiii_e >= min_snr) &
              (sii_f / sii_e >= min_snr) & (han_f / han_e >= min_snr) &
              (hb_f > 0) & (oiii_f > 0) & (sii_f > 0) & (han_f > 0))
    meas = snr_ok & np.isfinite(x_true) & np.isfinite(y_true) & (z_all >= z_min) & ds.valid.any(axis=1)
    idx_meas = np.where(meas)[0]
    print(f"[bpt] measured-line galaxies (z>=1, S/N>=3): N={len(idx_meas):,}")
    if len(idx_meas) < 100:
        print("  too few measured-line galaxies — skipping BPT")
        return {"N": int(len(idx_meas)), "skipped": True}

    # Kewley+01 [SII] demarcation → SF vs AGN labels
    def kewley01_sii(x):
        x = np.asarray(x, float); y = np.full(x.shape, np.inf)
        m = x < 0.32; y[m] = 0.72 / (x[m] - 0.32) + 1.30
        return y
    agn = ((x_true >= 0.32) | (y_true > kewley01_sii(x_true))).astype(np.float32)

    tok, V = encode_tokens(model, ds.flux, ds.wavelength, ds.valid, idx_meas, device)
    d = model.hparams.embed_dim
    xy = np.stack([x_true[idx_meas], y_true[idx_meas]], 1).astype(np.float32)
    agn_m = agn[idx_meas]
    z_m = z_all[idx_meas]

    # Regressor: predict (x, y), 80/20 train/val with early stopping on val.
    n = len(idx_meas)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    tr, va = perm[:int(0.8 * n)], perm[int(0.8 * n):]
    xy_va = xy[va.numpy()]
    xy_pred = train_regression(tok[tr], V[tr], torch.from_numpy(xy[tr]),
                               tok[va], V[va], xy_va, d, device, n_out=2, seed=seed)
    mae = np.abs(xy_pred - xy_va).mean(0)
    print(f"  regressor val MAE:  x={mae[0]:.3f}  y={mae[1]:.3f}")

    # Classifier: SF vs AGN, trained on the lowest-z 40% anchor, 80/20 + early stop.
    z_cut = float(np.quantile(z_m, 0.40))
    ia = np.where(z_m <= z_cut)[0]
    pA = torch.randperm(len(ia), generator=torch.Generator().manual_seed(7)).numpy()
    sp = max(1, int(0.8 * len(ia)))
    trA, valA = ia[pA[:sp]], ia[pA[sp:]]
    if len(valA) == 0:
        valA = trA
    logit = train_classifier(tok[trA], V[trA], agn_m[trA], tok[valA], V[valA], agn_m[valA],
                             d, device, seed=seed + 1)
    acc = float(((logit > 0).astype(int) == agn_m[valA].astype(int)).mean())
    print(f"  classifier anchor-val accuracy (z<={z_cut:.2f}): {acc:.1%}  "
          f"(AGN frac={agn_m[ia].mean():.1%}, N_anchor={len(ia):,})")
    return {"N": int(n), "regressor_mae_x": float(mae[0]), "regressor_mae_y": float(mae[1]),
            "classifier_anchor_acc": acc, "anchor_z_cut": z_cut, "anchor_agn_frac": float(agn_m[ia].mean())}


# ── plotting ─────────────────────────────────────────────────────────────────

def _plot_redshift(z_true, z_pred, m, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dz = (z_pred - z_true) / (1 + z_true)
    lim = (0, z_true.max() * 1.05)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1], "hspace": 0.06})
    ax.scatter(z_true, z_pred, s=4, alpha=0.35, color="#242222", edgecolors="none")
    ax.plot(lim, lim, "k-", lw=0.9); ax.set_ylim(lim); ax.set_ylabel(r"$z_{\rm pred}$")
    ax.set_title("SpecML redshift (frozen encoder + attn-pool probe)")
    ax.text(0.04, 0.95, f"$\\sigma_{{NMAD}}$={m['sigma_nmad']:.3f}\nR²={m['r2']:.3f}\n"
            f"cat>0.15={m['catastrophic_0p15']:.1%}\nN={len(z_true):,}",
            transform=ax.transAxes, va="top")
    axr.scatter(z_true, dz, s=4, alpha=0.35, color="#242222", edgecolors="none")
    for h in (0.0, 0.05, -0.05):
        axr.axhline(h, color="firebrick", lw=0.9, ls="--" if h else "-")
    axr.set_xlim(lim); axr.set_ylim(-0.15, 0.15)
    axr.set_xlabel(r"$z_{\rm true}$"); axr.set_ylabel(r"$\Delta z/(1+z)$")
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SpecML downstream evaluation")
    p.add_argument("--ckpt", required=True, help="Path to a SpecML .ckpt")
    p.add_argument("--catalog", default=DEFAULT_CATALOG)
    p.add_argument("--task", default="all", choices=["all", "redshift", "bpt"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Probe seeds for the redshift task (multi-seed mean).")
    p.add_argument("--device", default=None)
    p.add_argument("--outdir", default=None, help="Defaults to <ckpt_dir>/eval/")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "eval")
    os.makedirs(outdir, exist_ok=True)

    model = load_specml(args.ckpt, device=device)
    print(f"loaded {args.ckpt}  (embed_dim={model.hparams.embed_dim}, "
          f"patch_size={model.hparams.patch_size}, overlap={model.hparams.overlap})")

    ds = SpecMLDataset()  # raw flux / wave / valid + catalog_index
    cat = Table.read(args.catalog, format="ascii")
    cat = cat[cat["grating"] == "PRISM"][ds.catalog_index]  # align to dataset order
    assert len(cat) == len(ds), f"catalog/spectra misaligned: {len(cat)} vs {len(ds)}"

    results = {"ckpt": os.path.abspath(args.ckpt), "hparams": dict(model.hparams)}
    if args.task in ("all", "redshift"):
        results["redshift"] = eval_redshift(model, ds, cat, device, args.seeds, outdir)
    if args.task in ("all", "bpt"):
        results["bpt"] = eval_bpt(model, ds, cat, device, seed=1, outdir=outdir)

    out_json = os.path.join(outdir, "eval_metrics.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
