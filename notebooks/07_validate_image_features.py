"""
STAI-X Challenge 2026 — Step 7: validate the image features on the team tables.

Confirms that outputs/image_features.csv (from 06) merges cleanly onto the
team's modeling data and that it adds predictive value — using the SAME
validation engine as notebooks 04/05 (statx_helpers.cross_val_by_period), so the
numbers are directly comparable.

Validation: 5-fold GroupKFold by period_id (cross_val_by_period). Every period
is evaluated once, no period spans train/test, and Dataset B/C weather
imputation is re-fit inside each fold (no cross-fold leak). With only 77 periods
this is a lower-variance estimate than a single 80/20 holdout.

The image features carry no per-fold statistics (each is computed from one image
independently in notebook 06), so merging them onto the raw table before CV is
leak-free. For each weather variant (Dataset A/B/C) we compare the team's
Model 3 (HistGradientBoosting):
    A) team covariates only
    B) team covariates + image features

Run (from repo root):
    python notebooks/07_validate_image_features.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from statx_helpers import DATASET_SPECS, cross_val_by_period

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

KEYS = ["period_id", "jurisdiction"]
MODEL = "Model 3 - HistGradientBoosting"   # isolate the image effect on one model
N_SPLITS = 5
IMAGE_COLS = [
    "img_bg_ratio", "img_density_mean", "img_density_std", "img_density_max",
    "img_density_p90", "img_high_dens_frac", "img_top10_share", "img_spatial_spread",
]


def hgb_row(raw_df: pd.DataFrame, method: str, label: str) -> dict:
    """5-fold GroupKFold via the shared engine; keep only the HGB row."""
    rows = cross_val_by_period(raw_df, method, label, n_splits=N_SPLITS)
    return next(r for r in rows if r["model"] == MODEL)


def main() -> None:
    raw = pd.read_csv(OUT / "train_universal_merged.csv")

    feats = pd.read_csv(OUT / "image_features.csv")
    feats = feats[feats["split"] == "train"].drop(columns="split")
    raw_img = raw.merge(feats, on=KEYS, how="left")

    # Merge sanity: every covariate row should match exactly one image.
    miss = raw_img[IMAGE_COLS].isna().all(axis=1).sum()
    print(f"Merge check: {len(raw_img)} rows | unmatched: {miss} "
          f"({'OK' if miss == 0 else 'CHECK KEYS'})")

    print(f"\n{N_SPLITS}-fold GroupKFold by period (shared cross_val_by_period; "
          f"team Model 3)")
    print("Only difference between rows = the 8 image features\n")
    print(f"  {'dataset':<32}{'features':<16}{'RMSE':>9}{'MAE':>9}{'R2':>9}")
    print("  " + "-" * 73)

    for dataset_name, spec in DATASET_SPECS.items():
        method, label = spec["method"], dataset_name
        cov = hgb_row(raw, method, label)
        img = hgb_row(raw_img, method, label)
        print(f"  {label:<32}{'covariates':<16}"
              f"{cov['RMSE']:>9.4f}{cov['MAE']:>9.4f}{cov['R2']:>9.4f}")
        print(f"  {label:<32}{'+ image':<16}"
              f"{img['RMSE']:>9.4f}{img['MAE']:>9.4f}{img['R2']:>9.4f}")
        d_mae = (img['MAE'] - cov['MAE']) / cov['MAE'] * 100
        print(f"  {'':<32}{'MAE change':<16}{'':>9}{d_mae:>8.1f}%{'':>9}\n")


if __name__ == "__main__":
    main()
