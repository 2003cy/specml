# SpecML — masked-patch spectral pre-training

Refactor of the original `SpecML.py` / `Tokeniser.py` / `Training.py` into a
small package, structured after `ssl_outthere/encoder_spectrum/LowResPT`. The
pre-training recipe (architecture, normalisation, patch tokenisation, random
masking, MSE loss, warmup + SGDR-cosine schedule) is unchanged.

```
SpecML/
├── trainer.py            # LightningCLI entrypoint
├── specml_pt.yaml        # config (model / data / trainer)
├── callbacks.py          # EpochPrinter, PlotMetrics
├── visualize_metrics.py  # metrics.png from the CSVLogger CSV
├── model/
│   ├── modules.py         # SpectralAttention, SpectralBlock
│   └── specml.py          # SpecML(LightningModule): arch + preprocessing + train loop
└── data/
    ├── dataset.py         # SpecMLDataset (download FITS once → raw flux/wave/valid)
    └── datamodule.py      # SpecMLDataModule (split + dataloaders)
```

## Train

```bash
python trainer.py fit --config specml_pt.yaml
python trainer.py fit --config specml_pt.yaml --trainer.devices=[0,1]
python trainer.py fit --config specml_pt.yaml --trainer.logger.init_args.name=my_run
```

Outputs land in `output/<run_name>/version_<k>/`:
`metrics.csv`, `metrics.png` and `checkpoints/*.ckpt`.

The FITS is downloaded once into `SpecML/data/` and read from disk thereafter.


## Load a trained model (raw-data API)

```python
from model.specml import SpecML, load_specml
model = load_specml("output/my_run/version_0/checkpoints/last.ckpt")  # or SpecML.load_from_checkpoint(...)
# config restored on model.hparams

# flux / wavelength / valid_mask are (B, L) tensors of RAW data — no manual
# normalisation needed; the model does it internally. (LowResPT-style dict API.)
emb = model.embed(flux, wavelength, valid_mask)
#   emb["pooled"]           (B, D)     global spectrum embedding
#   emb["tokens"]           (B, T, D)  per-token embeddings (for attention-pooling probes)
#   emb["token_valid_mask"] (B, T)

out = model.reconstruct(flux, wavelength, valid_mask)               # all-visible
out = model.reconstruct(flux, wavelength, valid_mask, masked=True, mask_ratio=0.5)
#   out["recon"]/["target"] (B, T, P+2), out["valid"]/["mask"] (B, T),
#   out["flux_norm"] (B, L), out["wave_token"] (B, T)
```
