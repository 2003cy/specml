"""Optuna objective for the SpecML err-weighting hyperparameter search.

One trial = one 100-epoch Lightning run of the err-weighted SpecML. Every
`probe.every_n_epochs` epochs a frozen-encoder redshift probe is trained and its
σ_NMAD is (a) reported to Optuna for pruning and (b) tracked; the trial's
objective is the BEST σ_NMAD over the run (minimize). val_loss is NOT used — it's
dominated by the irreducible noise floor (see memory/specml-err-weighting).

The dataset (raw flux/err/wave) is built ONCE and reused across trials (patch
params are fixed), so trials don't re-read the 978 MB FITS each time.

Run via launch_study.py.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                     # the specml/ package
sys.path.insert(0, str(REPO))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import optuna
import pytorch_lightning as pl
import torch
import yaml
from astropy.table import Table
from pytorch_lightning.loggers import CSVLogger

from data.datamodule import SpecMLDataModule
from data.dataset import SpecMLDataset
from eval.run_eval import eval_redshift, DEFAULT_CATALOG
from model.specml import SpecML


# ─────────────────────────────────────────────────────────────────────────────
# Search space — edit freely. Fixed knobs live in configs/base.yaml.
# ─────────────────────────────────────────────────────────────────────────────
def suggest_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    return dict(
        # architecture (n_heads=8 fixed in base.yaml; these all divide by 8)
        embed_dim    = trial.suggest_categorical("embed_dim", [384, 512, 768]),
        n_layers     = trial.suggest_categorical("n_layers", [6, 8, 10, 12]),
        dropout      = trial.suggest_float("dropout", 0.0, 0.1),
        # masking + err-weighting strength (σ²_min is searched, per request)
        mask_ratio            = trial.suggest_float("mask_ratio", 0.3, 0.7),
        err_weight_sigma_min  = trial.suggest_float("err_weight_sigma_min", 0.1, 1.0, log=True),
        # optimizer / schedule
        lr            = trial.suggest_float("lr", 1e-4, 5e-4, log=True),
        weight_decay  = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True),
        warmup_steps  = trial.suggest_int("warmup_steps", 1000, 3000),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Redshift-probe objective + Optuna pruning, every N epochs
# ─────────────────────────────────────────────────────────────────────────────
class OptunaRedshiftProbe(pl.Callback):
    """Every `every_n_epochs`, train a frozen-encoder redshift probe on the live
    model, report σ_NMAD to Optuna (for pruning), and track the best so far.

    Encoder is forwarded under no_grad (frozen); only the probe trains. The
    global RNG is saved/restored so probing can't perturb the training mask
    stream, and validation inference_mode is disabled so the probe can backprop.
    """

    def __init__(self, trial, ds, cat, every_n_epochs=20, seeds=(0, 1)):
        self.trial = trial
        self.ds = ds
        self.cat = cat
        self.every = every_n_epochs
        self.seeds = list(seeds)
        self.best = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every != 0:
            return

        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.inference_mode(False), torch.enable_grad():
                res = eval_redshift(pl_module, self.ds, self.cat, pl_module.device,
                                    self.seeds, outdir=None)
            sn = res["mean"]["sigma_nmad"]
            self.best = min(self.best, sn)
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {"ds_z_sigma_nmad": sn,
                     "ds_z_catastrophic": res["mean"]["catastrophic_0p15"],
                     "ds_z_r2": res["mean"]["r2"]},
                    step=trainer.global_step)
            print(f"  [trial {self.trial.number}] epoch {trainer.current_epoch:3d}  "
                  f"σ_NMAD={sn:.4f}  (best={self.best:.4f})", flush=True)
            self.trial.report(sn, step=trainer.current_epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at epoch {trainer.current_epoch} (σ_NMAD={sn:.4f})")
        finally:
            if was_training:
                pl_module.train()
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────
def run_trial(trial, base_cfg, ds, cat, study_dir, max_epochs=100):
    hp = suggest_hparams(trial)
    model_cfg = {**base_cfg["model"], **hp}

    # Fresh DataModule per trial (so the trainer owns the data-loader lifecycle),
    # but inject the already-built dataset so the 978 MB FITS isn't re-read.
    dm = SpecMLDataModule(**base_cfg["data"])
    dm.dataset = ds

    trial_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "hparams.yaml").write_text(yaml.safe_dump(model_cfg, sort_keys=False))

    probe_cfg = base_cfg.get("probe", {})
    probe_cb = OptunaRedshiftProbe(
        trial, ds, cat,
        every_n_epochs=probe_cfg.get("every_n_epochs", 20),
        seeds=probe_cfg.get("seeds", [0, 1]),
    )

    try:
        model = SpecML(**model_cfg)
        # Precompute total optimisation steps (cosine schedule needs it; manual
        # trainer.fit can't use estimated_stepping_batches safely — see model).
        n_train = int(len(ds) * base_cfg["data"].get("train_val_split", 0.9))
        steps_per_epoch = max(1, math.ceil(n_train / base_cfg["data"].get("batch_size", 256)))
        model._total_steps_override = steps_per_epoch * max_epochs

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=1,
            precision="32",
            gradient_clip_val=0.5,
            logger=CSVLogger(save_dir=trial_dir, name="", version=""),
            callbacks=[probe_cb],
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=20,
        )
        trainer.fit(model, datamodule=dm)

        (trial_dir / "result.json").write_text(json.dumps({
            "trial_number": trial.number,
            "best_sigma_nmad": probe_cb.best,
            "hparams": hp,
            "epochs_run": trainer.current_epoch,
        }, indent=2))
        return probe_cb.best

    except optuna.TrialPruned:
        (trial_dir / "result.json").write_text(json.dumps({
            "trial_number": trial.number, "pruned": True,
            "best_sigma_nmad": probe_cb.best, "hparams": hp}, indent=2))
        raise
    except (RuntimeError, ValueError) as e:           # OOM / NaN → fail, keep worker alive
        (trial_dir / "FAILED.txt").write_text(f"{e}\n{traceback.format_exc()}")
        return float("inf")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
def load_base_cfg(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_data(base_cfg, catalog_path=None):
    """Build the raw dataset ONCE + aligned redshift labels (reused by all trials)."""
    ds = SpecMLDataset(min_std=base_cfg["data"].get("min_std", 0.0))
    cat = Table.read(catalog_path or DEFAULT_CATALOG, format="ascii")
    cat = cat[cat["grating"] == "PRISM"][ds.catalog_index]
    assert len(cat) == len(ds), f"catalog/spectra misaligned: {len(cat)} vs {len(ds)}"
    return ds, cat


def make_objective(base_cfg_path: Path, study_dir: Path, max_epochs: int = 100):
    base_cfg = load_base_cfg(base_cfg_path)
    ds, cat = build_data(base_cfg)       # FITS read ONCE, dataset reused by every trial
    def _objective(trial: optuna.Trial) -> float:
        return run_trial(trial, base_cfg, ds, cat, study_dir, max_epochs=max_epochs)
    return _objective
