"""SpecML pre-training dataset.

Reads the DJA PRISM spectra and applies the exact same valid-pixel selection,
flux conversion (f_nu → f_lambda via /λ²) and near-flat filtering as the
original Tokeniser.py — but stops there. Normalisation, tokenisation and the
wavelength positional encoding now live on the SpecML model
(`model.preprocess`), so this dataset returns RAW per-spectrum tensors and the
model does the rest. Kept deliberately minimal.
"""

import os
import shutil
import urllib.request

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import Dataset

# Same source the original Tokeniser.py used.
DEFAULT_FITS_URL = (
    "https://s3.amazonaws.com/msaexp-nirspec/extractions/"
    "dja_msaexp_emission_lines_v4.5.prism_spectra.fits"
)
# Download target: kept inside this package's data/ folder so the FITS lives
# next to the code that reads it and is downloaded only once.
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FITS_PATH = os.path.join(_DATA_DIR, "dja_msaexp_emission_lines_v4.5.prism_spectra.fits")


def ensure_local_fits(fits_path: str = DEFAULT_FITS_PATH, url: str = DEFAULT_FITS_URL) -> str:
    """Return a local path to the PRISM spectra FITS, downloading it once if missing.

    The file is fetched to ``<fits_path>.part`` first and only renamed into place
    on success, so an interrupted download never leaves a corrupt file behind.
    """
    if os.path.exists(fits_path):
        return fits_path
    os.makedirs(os.path.dirname(fits_path), exist_ok=True)
    tmp = fits_path + ".part"
    print(f"Downloading PRISM spectra FITS -> {fits_path}\n  from {url}", flush=True)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    os.replace(tmp, fits_path)
    print(f"Downloaded {os.path.getsize(fits_path) / 1e6:.1f} MB", flush=True)
    return fits_path


class SpecMLDataset(Dataset):
    """Raw DJA PRISM spectra for masked-patch pre-training.

    Args:
        fits_path:  Local path to the PRISM spectra FITS. If it doesn't exist it
                    is downloaded once from ``fits_url`` (default DJA v4.5 on S3)
                    into this package's data/ folder.
        fits_url:   Source URL used only when ``fits_path`` is missing.
        min_std:    Drop spectra whose raw flux std <= this (near-flat spectra).

    Each item is a dict of (L,) tensors: ``flux`` (f_lambda), ``wavelength`` (μm)
    and ``valid_mask`` (bool). All spectra share the same wavelength grid.
    """

    def __init__(self, fits_path: str = DEFAULT_FITS_PATH, fits_url: str = DEFAULT_FITS_URL,
                 min_std: float = 0.0):
        # Ensure the FITS lives locally (download once), then read from disk.
        fits_path = ensure_local_fits(fits_path, fits_url)
        data = Table.read(fits_path)

        # Filter for valid wavelengths / spectra (the data table is transposed:
        # rows index wavelength, columns index spectra).
        valid_w = np.any(data["valid"], 1)
        valid_spectrum = np.any(data["valid"], 0)

        w = np.asarray(data["wave"][valid_w], dtype=np.float32)            # (L,)
        f = data["flux"][np.ix_(valid_w, valid_spectrum)].T / (w ** 2)     # f_lambda, (B, L)
        e = data["err"][np.ix_(valid_w, valid_spectrum)].T / (w ** 2)      # per-pixel error, same units
        dq = data["valid"][np.ix_(valid_w, valid_spectrum)].T              # (B, L) per-pixel validity

        # Drop near-flat / constant spectra.
        spec_std = np.std(f, axis=1)
        keep = spec_std > min_std

        # Non-finite / non-positive error → +inf so its inverse-variance weight
        # is ~0 (a bad-error pixel contributes nothing to the weighted loss).
        e = np.asarray(e, dtype=np.float32)
        e[~np.isfinite(e) | (e <= 0)] = np.inf

        self.flux = np.ascontiguousarray(f[keep], dtype=np.float32)        # (B, L)
        self.err = np.ascontiguousarray(e[keep], dtype=np.float32)         # (B, L)
        self.valid = np.ascontiguousarray(np.asarray(dq)[keep], dtype=bool)  # (B, L)
        self.wavelength = w                                                # (L,) shared grid

        # Row index into the PRISM-filtered catalogue for each kept spectrum, so
        # downstream eval can align labels: equivalent to the notebook's
        # `catalog[grating==PRISM][valid_spectrum][valid_spectra]` selection.
        self.catalog_index = np.where(valid_spectrum)[0][keep]            # (B,)

        print(
            f"SpecMLDataset: {self.flux.shape[0]} spectra, {self.flux.shape[1]} pixels "
            f"({self.wavelength[0]:.3f}–{self.wavelength[-1]:.3f} µm)"
        )

    def __len__(self) -> int:
        return self.flux.shape[0]

    def __getitem__(self, idx):
        return {
            "flux": torch.from_numpy(self.flux[idx]),
            "err": torch.from_numpy(self.err[idx]),
            "wavelength": torch.from_numpy(self.wavelength.copy()),
            "valid_mask": torch.from_numpy(self.valid[idx]),
        }
