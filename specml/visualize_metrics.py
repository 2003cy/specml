"""Render a training-metrics figure (metrics.png) from the CSVLogger CSV.

Used two ways:
  * imported by the ``PlotMetrics`` callback (callbacks.py), which refreshes
    ``<log_dir>/metrics.png`` every few epochs during training;
  * standalone:  ``python visualize_metrics.py <metrics.csv> [out.png]``.

SpecML logs a per-step ``train_loss`` + ``lr`` and a per-epoch ``val_loss``;
this lays them out as masked-recon loss (train vs val) and the LR schedule.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_SMOOTH = 80  # rolling window for the (noisy) per-step train curve
_SKIP_FIRST = 100  # drop the first logging points (unstable warmup régime);
                   # clamped to <=20% of the run below.

_C_TRAIN = "#1f77b4"
_C_VAL = "#d62728"


def _style() -> None:
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "font.size": 9,
        "legend.frameon": False,
    })


def plot_metrics(csv_path: str, out_path: str | None = None, run_name: str | None = None,
                 skip_first: int = _SKIP_FIRST) -> str | None:
    """Write a metrics figure next to ``csv_path`` (or to ``out_path``). Returns the path."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, OSError):
        return None
    if "step" not in df.columns or "train_loss" not in df.columns:
        return None

    # Lightning logs train and val in separate rows; forward-fill epoch so val rows
    # inherit theirs, and derive a step->epoch scale for the top axis.
    if "epoch" in df.columns:
        df["epoch"] = df["epoch"].ffill()
        ref = df.dropna(subset=["epoch", "step"])
        x_to_epoch = (float(ref["epoch"].iloc[-1]) / float(ref["step"].iloc[-1])
                      if len(ref) and ref["step"].iloc[-1] else 0.0)
    else:
        x_to_epoch = 0.0

    out_path = out_path or os.path.join(os.path.dirname(csv_path), "metrics.png")
    tr_all = df.dropna(subset=["train_loss"]).sort_values("step")
    skip = min(skip_first, len(tr_all) // 5)  # never drop more than ~20% (short runs)
    tr = tr_all.iloc[skip:]
    va = df.dropna(subset=["val_loss"]).sort_values("step") if "val_loss" in df.columns else df.iloc[:0]
    if len(tr) == 0:
        return None
    x_tr = tr["step"].to_numpy()

    # Downstream-eval rows (logged sparsely by the DownstreamEval callback).
    has_ds = "ds_z_sigma_nmad" in df.columns and df["ds_z_sigma_nmad"].notna().any()

    _style()
    ncol = 3 if has_ds else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.5 * ncol, 4.8), constrained_layout=True)

    # ── Panel 0: masked-reconstruction loss (train vs val) ──
    ax = axes[0]
    ax.plot(x_tr, tr["train_loss"], color=_C_TRAIN, alpha=0.30, lw=0.9, label="train")
    ax.plot(x_tr, tr["train_loss"].rolling(_SMOOTH, min_periods=5).mean(),
            color=_C_TRAIN, lw=1.4, alpha=0.95, label="train (smoothed)")
    if len(va):
        ax.plot(va["step"], va["val_loss"], "o-", color=_C_VAL, ms=4, lw=1.2, label="val")
    if (tr["train_loss"] > 0).any():
        ax.set_yscale("log")
    ax.set_title("masked-recon loss (train_loss / val_loss) [mse]", pad=22)
    ax.legend(loc="best", fontsize=8)

    # ── Panel 1: LR schedule ──
    ax = axes[1]
    if "lr" in tr.columns and tr["lr"].notna().any():
        ax.plot(x_tr, tr["lr"].to_numpy(dtype=float), color="#2ca02c", lw=1.5, label="lr")
    ax.set_title("lr schedule (lr)", pad=22)
    ax.legend(loc="best", fontsize=8)

    # ── Panel 2 (optional): downstream redshift probe over training ──
    if has_ds:
        ax = axes[2]
        ds = df.dropna(subset=["ds_z_sigma_nmad"]).sort_values("step")
        ax.plot(ds["step"], ds["ds_z_sigma_nmad"], "o-", color="#9467bd", ms=4, lw=1.3,
                label="σ_NMAD")
        if "ds_z_catastrophic" in ds.columns:
            ax.plot(ds["step"], ds["ds_z_catastrophic"], "s--", color="#8c564b", ms=3, lw=1.0,
                    label="catastrophic >0.15")
        ax.set_title("downstream redshift probe (frozen) — lower=better", pad=22)
        ax.legend(loc="best", fontsize=8)
        if "ds_z_r2" in ds.columns and ds["ds_z_r2"].notna().any():
            axr2 = ax.twinx()
            axr2.plot(ds["step"], ds["ds_z_r2"], "^:", color="#2ca02c", ms=3, lw=1.0, label="R²")
            axr2.set_ylabel("R²", fontsize=8); axr2.tick_params(labelsize=7)

    for ax in axes:
        ax.set_xlabel("global step", fontsize=9)
        ax.tick_params(labelsize=8)
        if x_to_epoch:  # top axis: same range in epochs
            sec = ax.secondary_xaxis("top", functions=(lambda x: x * x_to_epoch,
                                                        lambda e: e / x_to_epoch))
            sec.set_xlabel("epoch", fontsize=8)
            sec.tick_params(labelsize=7)

    epoch = int(df["epoch"].max()) if "epoch" in df.columns and df["epoch"].notna().any() else -1
    step = int(tr["step"].iloc[-1])
    vlast = float(va["val_loss"].iloc[-1]) if len(va) else float("nan")
    name = run_name or os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    fig.suptitle(f"{name}    |    epoch {epoch}  ·  step {step}  ·  val_loss {vlast:.4f}",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python visualize_metrics.py <metrics.csv> [out.png]")
    written = plot_metrics(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"wrote {written}")
