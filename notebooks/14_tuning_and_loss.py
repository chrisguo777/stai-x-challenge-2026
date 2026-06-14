"""
STAI-X Challenge 2026 — Step 14: MAE-aligned loss, light tuning, ensemble weight.

Quick, principled wins, validated on the same 5-fold GroupKFold-by-period
(scored-only MAE). Set MULTITASK below to match notebook 13's verdict (train on
all 8 categories vs the 3 scored).

Experiments:
  1. HistGB loss: default squared_error vs absolute_error (matches the MAE metric)
  2. light tuning of the better HistGB
  3. ensemble blend weight sweep (HistGB vs RandomForest) on OOF predictions

Run (from repo root):
    python notebooks/14_tuning_and_loss.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from statx_helpers import make_dataset, build_preprocessor, TARGET_COL, PERIOD_COL

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
KEYS = ["period_id", "jurisdiction"]
METHOD = "similar_state"
N_SPLITS = 5
SCORED = ["all_drugs", "all_opioids", "all_stimulants"]
MULTITASK = False                 # notebook 13 verdict: multitask hurt, use scored-only
IMAGE_COLS = [
    "img_bg_ratio", "img_density_mean", "img_density_std", "img_density_max",
    "img_density_p90", "img_high_dens_frac", "img_top10_share", "img_spatial_spread",
]
DROP_FROM_X = ["row_id", TARGET_COL, "state_doh_release"]


def load_raw() -> pd.DataFrame:
    target = pd.read_csv(ROOT / "train" / "dose_sys_train.csv")
    if not MULTITASK:
        target = target[target["overdose_category"].isin(SCORED)]
    cov = pd.read_csv(ROOT / "train" / "covariates.csv")
    img = pd.read_csv(OUT / "image_features.csv")
    img = img[img["split"] == "train"][KEYS + IMAGE_COLS]
    return (target.merge(cov, on=KEYS, how="left").merge(img, on=KEYS, how="left")
            .dropna(subset=[TARGET_COL]).reset_index(drop=True))


def oof_predictions(raw: pd.DataFrame, estimators: dict):
    """5-fold OOF preds for each estimator, plus the held-out scored truth."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    periods = raw[PERIOD_COL].to_numpy()
    y_all, oof = [], {name: [] for name in estimators}
    for tr_idx, te_idx in gkf.split(raw, raw[TARGET_COL], periods):
        train_raw = raw.iloc[tr_idx]
        test_scored = raw.iloc[te_idx]
        test_scored = test_scored[test_scored["overdose_category"].isin(SCORED)]
        train_ds = make_dataset(train_raw, METHOD)
        test_ds = make_dataset(test_scored, METHOD, reference=train_raw)
        X_tr = train_ds.drop(columns=DROP_FROM_X, errors="ignore")
        y_tr = train_ds[TARGET_COL].to_numpy()
        X_te = test_ds.drop(columns=DROP_FROM_X, errors="ignore").reindex(columns=X_tr.columns)
        y_all.append(test_ds[TARGET_COL].to_numpy())
        for name, est in estimators.items():
            pipe = Pipeline([("pre", build_preprocessor(X_tr)), ("m", clone_est(est))])
            pipe.fit(X_tr, y_tr)
            oof[name].append(np.clip(pipe.predict(X_te), 0, None))
    y = np.concatenate(y_all)
    return y, {name: np.concatenate(v) for name, v in oof.items()}


def clone_est(est):
    from sklearn.base import clone
    return clone(est)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def main() -> None:
    raw = load_raw()
    print(f"MULTITASK={MULTITASK} | rows={len(raw)} | periods={raw[PERIOD_COL].nunique()}\n")

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
    estimators = {
        "HGB squared": HistGradientBoostingRegressor(random_state=42),
        "HGB absolute": HistGradientBoostingRegressor(loss="absolute_error", random_state=42),
        "HGB abs tuned": HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=600,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=42),
        "RandomForest": rf,
    }
    y, oof = oof_predictions(raw, estimators)

    print("  single-model scored MAE (5-fold OOF):")
    for name in estimators:
        print(f"    {name:<16}{mae(y, oof[name]):.4f}")

    # Ensemble weight sweep: w*HGB + (1-w)*RF, using the best HGB variant.
    best_hgb = min(["HGB squared", "HGB absolute", "HGB abs tuned"], key=lambda n: mae(y, oof[n]))
    print(f"\n  best HGB variant: {best_hgb}")
    print("  blend  w*{HGB} + (1-w)*RF:")
    best = (None, 1e9)
    for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
        m = mae(y, w * oof[best_hgb] + (1 - w) * oof["RandomForest"])
        flag = "  <--" if m < best[1] else ""
        if m < best[1]:
            best = (w, m)
        print(f"    w={w:.1f}  MAE={m:.4f}{flag}")
    print(f"\n  best blend: w={best[0]:.1f} ({best[1]:.4f})  [baseline single HGB 1.7247 / ens 1.7125]")


if __name__ == "__main__":
    main()
