"""DataModule for SpecML masked-patch pre-training.

Builds the raw-spectrum dataset once and splits it into train/val. All spectra
share the same wavelength grid, so the default collate stacks them with no
padding; normalisation + tokenisation happen on the model. Directly
instantiable by LightningCLI from the YAML `data:` section.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import SpecMLDataset, DEFAULT_FITS_PATH, DEFAULT_FITS_URL


class SpecMLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fits_path: str = DEFAULT_FITS_PATH,
        fits_url: str = DEFAULT_FITS_URL,
        min_std: float = 0.0,
        batch_size: int = 256,
        num_workers: int = 4,
        train_val_split: float = 0.9,
        split_seed: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: str = None) -> None:
        # Idempotent on the split. If a dataset was injected (e.g. shared across
        # Optuna trials via `dm.dataset = ...` to avoid re-reading the FITS), we
        # keep it and only build the train/val split here.
        if self.train_ds is not None:
            return
        if self.dataset is None:
            self.dataset = SpecMLDataset(
                fits_path=self.hparams.fits_path,
                fits_url=self.hparams.fits_url,
                min_std=self.hparams.min_std,
            )
        n_train = int(len(self.dataset) * self.hparams.train_val_split)
        n_val = len(self.dataset) - n_train
        self.train_ds, self.val_ds = random_split(
            self.dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.hparams.split_seed),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )
