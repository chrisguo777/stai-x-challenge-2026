"""
STAI-X Challenge 2026 — Step 12: validate a HistGB + RandomForest ensemble.

Before committing a submission to the ensemble, confirm under the project's
5-fold GroupKFold-by-period that averaging the two tree models actually beats
the single best model (HistGB, MAE 1.725 on Dataset C + image).

Setup: Universal + Dataset C + image features. Weather imputation is re-fit
inside each fold (no cross-fold leak); image features are per-row independent so
merging them before CV is leak-free.

Run (from repo root):
    python notebooks/12_ensemble_validation.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from statx_helpers import make_dataset, build_preprocessor, evaluate_model, TARGET_COL, PERIOD_COL

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
KEYS = ["period_id", "jurisdiction"]
METHOD = "similar_state"        # Dataset C
N_SPLITS = 5
IMAGE_COLS = [
    "img_bg_ratio", "img_density_mean", "img_density_std", "img_density_max",
    "img_density_p90", "img_high_dens_frac", "img_top10_share", "img_spatial_spread",
]
DROP_FROM_X = ["row_id", TARGET_COL]


def make_models():
    return {
        "HistGB": HistGradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
    }


def main() -> None:
    raw = pd.read_csv(OUT / "train_universal_merged.csv")
    img = pd.read_csv(OUT / "image_features.csv")
    img = img[img["split"] == "train"][KEYS + IMAGE_COLS]
    raw = raw.merge(img, on=KEYS, how="left").dropna(subset=[TARGET_COL]).reset_index(drop=True)

    groups = raw[PERIOD_COL].to_numpy()
    gkf = GroupKFold(n_splits=N_SPLITS)

    scores = {"HistGB": [], "RandomForest": [], "Ensemble (avg)": []}
    for tr_idx, te_idx in gkf.split(raw, raw[TARGET_COL], groups):
        train_fold, test_fold = raw.iloc[tr_idx], raw.iloc[te_idx]
        train_ds = make_dataset(train_fold, METHOD)
        test_ds = make_dataset(test_fold, METHOD, reference=train_fold)

        X_tr = train_ds.drop(columns=DROP_FROM_X, errors="ignore")
        y_tr = train_ds[TARGET_COL].to_numpy()
        X_te = test_ds.drop(columns=DROP_FROM_X, errors="ignore")
        y_te = test_ds[TARGET_COL].to_numpy()

        preds = {}
        for name, est in make_models().items():
            pipe = Pipeline([("preprocess", build_preprocessor(X_tr)), ("model", est)])
            pipe.fit(X_tr, y_tr)
            preds[name] = np.clip(pipe.predict(X_te), 0, None)
            scores[name].append(evaluate_model(y_te, preds[name])[1])      # MAE

        ens = (preds["HistGB"] + preds["RandomForest"]) / 2
        scores["Ensemble (avg)"].append(evaluate_model(y_te, ens)[1])

    print("5-fold GroupKFold by period — Dataset C + image — MAE\n")
    print(f"  {'model':<20}{'MAE':>8}")
    print("  " + "-" * 28)
    for name, vals in scores.items():
        print(f"  {name:<20}{np.mean(vals):>8.4f}")
    best = min(scores, key=lambda k: np.mean(scores[k]))
    print(f"\n  best: {best} ({np.mean(scores[best]):.4f})")


if __name__ == "__main__":
    main()
