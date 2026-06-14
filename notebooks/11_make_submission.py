"""
STAI-X Challenge 2026 — Step 11: build the submission.

Final model (see README «最终方案与选型理由»):
    Universal + Dataset C (weather similar-state imputation) + image features
    + HistGradientBoosting (team Model 3).

Trains on ALL labeled training rows (no fold held out — folds were only for
selection), predicts the official val/ rows, and writes submission.csv at the
repo root with exactly the sample_submission row_ids.

Note: text features (09/10) are intentionally NOT included — they gave no gain
under 5-fold (slightly worse on trees).

Run (from repo root):
    python notebooks/11_make_submission.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from statx_helpers import make_dataset, build_preprocessor, TARGET_COL

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
KEYS = ["period_id", "jurisdiction"]
METHOD = "similar_state"          # Dataset C
IMAGE_COLS = [
    "img_bg_ratio", "img_density_mean", "img_density_std", "img_density_max",
    "img_density_p90", "img_high_dens_frac", "img_top10_share", "img_spatial_spread",
]
DROP_FROM_X = ["row_id", TARGET_COL]


def add_image(df: pd.DataFrame, split: str) -> pd.DataFrame:
    img = pd.read_csv(OUT / "image_features.csv")
    img = img[img["split"] == split][KEYS + IMAGE_COLS]
    return df.merge(img, on=KEYS, how="left")


def main() -> None:
    train_raw = pd.read_csv(OUT / "train_universal_merged.csv")
    val_raw = pd.read_csv(OUT / "val_universal_merged.csv")

    # Dataset C modeling tables (val weather imputed from TRAIN stats only).
    train_ds = add_image(make_dataset(train_raw, METHOD), "train")
    val_ds = add_image(make_dataset(val_raw, METHOD, reference=train_raw), "val")

    train_ds = train_ds.dropna(subset=[TARGET_COL])

    # Sanity: every val row must have matched its image features.
    miss = val_ds[IMAGE_COLS].isna().all(axis=1).sum()
    print(f"val rows: {len(val_ds)} | image-unmatched: {miss} "
          f"({'OK' if miss == 0 else 'CHECK KEYS'})")

    X_train = train_ds.drop(columns=DROP_FROM_X, errors="ignore")
    y_train = train_ds[TARGET_COL].to_numpy()
    X_val = val_ds.drop(columns=DROP_FROM_X, errors="ignore")
    assert list(X_train.columns) == list(X_val.columns), "train/val feature columns differ"

    model = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("model", HistGradientBoostingRegressor(random_state=42)),
    ])
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_val), 0, None)   # rates cannot be negative

    # Align to the official template: exactly its row_ids, in its order.
    sub = pd.read_csv(ROOT / "sample_submission.csv")[["row_id"]]
    pred_df = pd.DataFrame({"row_id": val_ds["row_id"].to_numpy(),
                            TARGET_COL: preds})
    out = sub.merge(pred_df, on="row_id", how="left")

    assert len(out) == 918, f"expected 918 rows, got {len(out)}"
    assert out["row_id"].is_unique
    assert out[TARGET_COL].notna().all(), "some submission rows have no prediction"

    out_path = ROOT / "submission.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out)} rows)")
    print(out[TARGET_COL].describe().round(3).to_string())
    print("\npreview:")
    print(out.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
