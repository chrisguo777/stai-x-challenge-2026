"""
STAI-X Challenge 2026 — Step 13: multi-task with the 5 non-scored categories.

dose_sys_train.csv has 8 overdose categories; we currently train on only the 3
scored ones. The other 5 (heroin, fentanyl, cocaine, methamphetamine,
benzodiazepine) are 2.7x more training rows, strongly correlated with the scored
targets. val/ has no target for any category, so they can't be features — but
they CAN be extra TRAINING rows (multi-task): train one universal model on all 8
categories (overdose_category as a feature), predict only the 3 scored.

This validates that idea with a fair, shared-fold comparison under the project's
5-fold GroupKFold-by-period, scoring ONLY the 3 scored categories on held-out
periods:

    baseline   : train on the 3 scored categories
    multi-task : train on all 8 categories

Both use Dataset C + image features and identical period folds. Adopt multi-task
only if it lowers scored MAE vs the 1.7125 ensemble / 1.7247 HistGB baseline.

Run (from repo root):
    python notebooks/13_multitask_categories.py
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
METHOD = "similar_state"          # Dataset C
N_SPLITS = 5
SCORED = ["all_drugs", "all_opioids", "all_stimulants"]
IMAGE_COLS = [
    "img_bg_ratio", "img_density_mean", "img_density_std", "img_density_max",
    "img_density_p90", "img_high_dens_frac", "img_top10_share", "img_spatial_spread",
]
DROP_FROM_X = ["row_id", TARGET_COL, "state_doh_release"]


def make_models():
    return {
        "HistGB": HistGradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
    }


def fit_predict(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> dict:
    """Train each model on train_raw, predict test_raw. Returns {model: preds}."""
    train_ds = make_dataset(train_raw, METHOD)
    test_ds = make_dataset(test_raw, METHOD, reference=train_raw)
    X_tr = train_ds.drop(columns=DROP_FROM_X, errors="ignore")
    y_tr = train_ds[TARGET_COL].to_numpy()
    X_te = test_ds.drop(columns=DROP_FROM_X, errors="ignore").reindex(columns=X_tr.columns)
    preds = {}
    for name, est in make_models().items():
        pipe = Pipeline([("pre", build_preprocessor(X_tr)), ("m", est)])
        pipe.fit(X_tr, y_tr)
        preds[name] = np.clip(pipe.predict(X_te), 0, None)
    preds["Ensemble"] = (preds["HistGB"] + preds["RandomForest"]) / 2
    return preds


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def main() -> None:
    target = pd.read_csv(ROOT / "train" / "dose_sys_train.csv")          # all 8 categories
    cov = pd.read_csv(ROOT / "train" / "covariates.csv")
    img = pd.read_csv(OUT / "image_features.csv")
    img = img[img["split"] == "train"][KEYS + IMAGE_COLS]

    raw = (target.merge(cov, on=KEYS, how="left")
                 .merge(img, on=KEYS, how="left")
                 .dropna(subset=[TARGET_COL]).reset_index(drop=True))

    periods = raw[PERIOD_COL].to_numpy()
    gkf = GroupKFold(n_splits=N_SPLITS)

    rows = {("baseline", m): [] for m in ["HistGB", "RandomForest", "Ensemble"]}
    rows.update({("multitask", m): [] for m in ["HistGB", "RandomForest", "Ensemble"]})
    per_cat = {("baseline", c): [] for c in SCORED}
    per_cat.update({("multitask", c): [] for c in SCORED})

    for tr_idx, te_idx in gkf.split(raw, raw[TARGET_COL], periods):
        train_all8 = raw.iloc[tr_idx]
        train_scored = train_all8[train_all8["overdose_category"].isin(SCORED)]
        test_scored = raw.iloc[te_idx]
        test_scored = test_scored[test_scored["overdose_category"].isin(SCORED)]
        y_te = test_scored[TARGET_COL].to_numpy()

        for cond, train_raw in [("baseline", train_scored), ("multitask", train_all8)]:
            preds = fit_predict(train_raw, test_scored)
            for m, p in preds.items():
                rows[(cond, m)].append(mae(y_te, p))
            # per-category MAE for the ensemble
            ens = preds["Ensemble"]
            for c in SCORED:
                mask = (test_scored["overdose_category"] == c).to_numpy()
                per_cat[(cond, c)].append(mae(y_te[mask], ens[mask]))

    print("5-fold GroupKFold by period — Dataset C + image — scored-only MAE\n")
    print(f"  {'condition':<12}{'HistGB':>9}{'RandomForest':>14}{'Ensemble':>10}")
    print("  " + "-" * 45)
    for cond in ["baseline", "multitask"]:
        print(f"  {cond:<12}"
              f"{np.mean(rows[(cond,'HistGB')]):>9.4f}"
              f"{np.mean(rows[(cond,'RandomForest')]):>14.4f}"
              f"{np.mean(rows[(cond,'Ensemble')]):>10.4f}")

    print("\n  per-category MAE (Ensemble):")
    print(f"  {'condition':<12}" + "".join(f"{c:>16}" for c in SCORED))
    for cond in ["baseline", "multitask"]:
        print(f"  {cond:<12}" + "".join(f"{np.mean(per_cat[(cond,c)]):>16.4f}" for c in SCORED))


if __name__ == "__main__":
    main()
