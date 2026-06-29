"""Launch / resume the SpecML err-weighting Optuna study.

    python launch_study.py --study-name ew1 --n-trials 40
    python launch_study.py --study-name ew1 --n-trials 40 --gpu 1   # second GPU, same DB

The study lives in studies/<name>/study.db (sqlite). Running this command again
with the same --study-name resumes it; running it in parallel on another GPU
(different --gpu) adds workers to the same study — sqlite handles the concurrency.

Objective = best redshift-probe σ_NMAD per trial (minimize); MedianPruner cuts
weak trials early using the per-20-epoch σ_NMAD reports.
"""

import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES for this worker")
    ap.add_argument("--base-config", type=str, default=str(HERE / "configs" / "base.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Pin GPU before importing torch (via optuna_train)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    from optuna_train import make_objective

    study_dir = HERE / "studies" / args.study_name
    (study_dir / "trials").mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir / 'study.db'}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",                       # σ_NMAD, lower is better
        sampler=TPESampler(seed=args.seed, multivariate=True, group=True,
                           n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=20),
        load_if_exists=True,
    )

    objective = make_objective(Path(args.base_config), study_dir, max_epochs=args.max_epochs)
    print(f"[{time.strftime('%H:%M:%S')}] study '{args.study_name}' gpu={args.gpu} "
          f"target {args.n_trials} trials → {storage}", flush=True)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True, catch=())

    print(f"\n=== best trial #{study.best_trial.number}  σ_NMAD={study.best_value:.4f} ===")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
