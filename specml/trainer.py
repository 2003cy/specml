"""LightningCLI entrypoint for SpecML masked-patch pre-training.

Usage:
    python trainer.py fit --config specml_pt.yaml
    python trainer.py fit --config specml_pt.yaml --trainer.devices=[0,1]

SpecML *is* the LightningModule (LowResPT-style) and owns its preprocessing
(normalisation + tokenisation + positional encoding), so the model and data
configs are fully decoupled — no argument linking is needed. Checkpoints are
native Lightning ``.ckpt`` files (ModelCheckpoint, configured in the YAML); load
them with ``SpecML.load_from_checkpoint`` / ``load_specml``.
"""

import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Ensure local model/ and data/ are importable when running from this directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pytorch_lightning.cli import LightningCLI

from data.datamodule import SpecMLDataModule
from model.specml import SpecML


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    LightningCLI(
        model_class=SpecML,
        datamodule_class=SpecMLDataModule,
        save_config_callback=None,
        seed_everything_default=42,
    )


if __name__ == "__main__":
    main()
