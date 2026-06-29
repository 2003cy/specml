"""Lightweight Lightning callbacks for SpecML training."""

import os

import torch
from pytorch_lightning.callbacks import Callback

from visualize_metrics import plot_metrics


class EpochPrinter(Callback):
    """Print one plain line per epoch instead of a live progress bar.

    A plain ``print`` with a newline renders cleanly when training is launched as
    a subprocess (``!python trainer.py ...`` in a notebook), where in-place
    progress-bar redraws pile up instead of overwriting.
    """

    @staticmethod
    def _fmt(metrics, key, nd=4):
        v = metrics.get(key)
        if v is None:
            return "  n/a "
        try:
            return f"{v.item():.{nd}f}"
        except AttributeError:
            return f"{float(v):.{nd}f}"

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        lr = m.get("lr")
        lr_s = f"{lr.item():.2e}" if lr is not None else "n/a"
        print(
            f"epoch {trainer.current_epoch:3d} | "
            f"train_loss={self._fmt(m, 'train_loss')} | "
            f"val_loss={self._fmt(m, 'val_loss')} | "
            f"lr={lr_s}",
            flush=True,
        )


class PlotMetrics(Callback):
    """Refresh ``<log_dir>/metrics.png`` from the CSVLogger CSV every few epochs.

    Fires on train-epoch end so train curves refresh during training. Wrapped so
    a plotting hiccup can never interrupt training.
    """

    PLOT_EVERY_N_EPOCHS = 2

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero or trainer.log_dir is None:
            return
        if trainer.current_epoch % self.PLOT_EVERY_N_EPOCHS != 0:
            return
        try:
            if trainer.logger is not None:
                trainer.logger.save()  # flush pending rows to metrics.csv
            csv_path = os.path.join(trainer.log_dir, "metrics.csv")
            run_name = os.path.basename(trainer.log_dir.rstrip("/"))
            plot_metrics(csv_path, os.path.join(trainer.log_dir, "metrics.png"), run_name)
        except Exception as e:  # never let plotting kill a run
            print(f"[PlotMetrics] skipped ({type(e).__name__}: {e})", flush=True)


class DownstreamEval(Callback):
    """Run the frozen-encoder downstream probes every N epochs, log to metrics.

    Uses the LIVE model (== latest weights) and the training DataModule's dataset
    (no FITS re-read). Trains a small attention-pooling probe on top of the
    frozen encoder — gradients never reach the encoder — and logs the headline
    downstream numbers (`ds_z_*`) so they land in metrics.csv next to the loss
    curves. The global RNG is saved/restored so probing never perturbs the
    training masking stream.
    """

    def __init__(self, catalog_path: str = None, every_n_epochs: int = 10,
                 seeds=(0,), run_bpt: bool = False):
        self.catalog_path = catalog_path
        self.every_n_epochs = every_n_epochs
        self.seeds = list(seeds)
        self.run_bpt = run_bpt
        self._cat = None  # aligned catalogue, loaded once

    def _labels(self, ds):
        if self._cat is None:
            from astropy.table import Table
            from eval.run_eval import DEFAULT_CATALOG
            path = self.catalog_path or DEFAULT_CATALOG
            cat = Table.read(path, format="ascii")
            self._cat = cat[cat["grating"] == "PRISM"][ds.catalog_index]
        return self._cat

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        ds = getattr(getattr(trainer, "datamodule", None), "dataset", None)
        if ds is None:
            return

        from eval.run_eval import eval_redshift, eval_bpt

        # Save RNG so the probe's seeding can't shift the training mask stream.
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = pl_module.training
        pl_module.eval()
        try:
            # This hook runs inside Lightning's validation inference_mode, which
            # blocks autograd — re-enable it so the probe can train (the encoder
            # itself is forwarded under no_grad inside encode_tokens).
            with torch.inference_mode(False), torch.enable_grad():
                cat = self._labels(ds)
                log = {}
                r = eval_redshift(pl_module, ds, cat, pl_module.device, self.seeds, outdir=None)
                log["ds_z_sigma_nmad"] = r["mean"]["sigma_nmad"]
                log["ds_z_catastrophic"] = r["mean"]["catastrophic_0p15"]
                log["ds_z_r2"] = r["mean"]["r2"]
                if self.run_bpt:
                    b = eval_bpt(pl_module, ds, cat, pl_module.device, seed=1, outdir=None)
                    if not b.get("skipped"):
                        log["ds_bpt_mae"] = 0.5 * (b["regressor_mae_x"] + b["regressor_mae_y"])
                        log["ds_bpt_acc"] = b["classifier_anchor_acc"]
            if trainer.logger is not None:
                trainer.logger.log_metrics(log, step=trainer.global_step)
            print(f"epoch {trainer.current_epoch:3d} | downstream: "
                  + "  ".join(f"{k}={v:.4f}" for k, v in log.items()), flush=True)
        except Exception as e:  # never let eval kill a run
            print(f"[DownstreamEval] skipped ({type(e).__name__}: {e})", flush=True)
        finally:
            if was_training:
                pl_module.train()
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)
