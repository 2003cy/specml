# SpecML err-weighting — Optuna hyperparameter search

One trial = one **100-epoch** run of the err-weighted SpecML
(`loss_weighting=invvar`). Every **20 epochs** a frozen-encoder **redshift probe**
is trained and its **σ_NMAD** is reported to Optuna; the trial's objective is the
**best σ_NMAD over the run** (minimize). `val_loss` is deliberately *not* the
objective — it's dominated by the irreducible noise floor.

## Run

```bash
cd specml/optuna
python launch_study.py --study-name ew1 --n-trials 40
# resume: same command again. parallel on another GPU: --gpu 1 (same study DB)
```

Study lives in `studies/<name>/study.db` (sqlite); per-trial artifacts
(`hparams.yaml`, `metrics.csv`, `result.json`) under `studies/<name>/trials/`.

## Inspect

```python
import optuna
s = optuna.load_study(study_name="ew1", storage="sqlite:///studies/ew1/study.db")
print(s.best_value, s.best_params)
s.trials_dataframe().to_csv("ew1.csv")
```

## Searched parameters (`suggest_hparams` in `optuna_train.py`)

| param | range |
|---|---|
| `err_weight_sigma_min` | loguniform [0.1, 1.0] |
| `mask_ratio` | [0.3, 0.7] |
| `lr` | loguniform [1e-4, 5e-4] |
| `weight_decay` | loguniform [1e-3, 5e-2] |
| `n_layers` | {6, 8, 10, 12} |
| `embed_dim` | {384, 512, 768} |
| `dropout` | [0.0, 0.1] |
| `warmup_steps` | int [1000, 3000] |

**Fixed** (in `configs/base.yaml`): `loss_weighting=invvar`, `patch_size=4`,
`overlap=2` (changing patch_size would entangle with σ²_min), `n_heads=8`,
`ffn_ratio=4`, `betas`, `min_lr`, batch/data settings.

## Notes

- The FITS is read **once** and the dataset is reused across trials.
- Pruning: `MedianPruner` cuts a trial whose σ_NMAD is worse than the median at
  the same epoch (after 10 startup trials, no prune before epoch 20).
- Probe uses 2 seeds (`probe.seeds` in base.yaml) to reduce objective noise; the
  encoder is frozen, only the attention-pool + MLP probe trains.
