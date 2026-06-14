"""
STAI-X Challenge 2026 — self-contained Code-Competition submission notebook.

This is the script that Kaggle re-runs on the hidden later-window test set, so it
must build EVERYTHING from the raw competition data — no dependency on the local
outputs/*.csv. It reproduces the full validated pipeline from scratch:

    Universal (one model, all 3 categories, overdose_category as a feature)
    + Dataset C weather imputation (similar-state -> period-median -> global,
      all computed from TRAIN only)
    + 8 MAT-density image features (computed inline from the PNGs)
    + 0.7/0.3 ensemble of tuned HistGradientBoosting (absolute_error loss) and
      RandomForest

Local 5-fold GroupKFold-by-period MAE: 1.7086 (tuned HistGB + 0.7/0.3 blend;
beats the 0.5/0.5 average 1.7125 and single default HistGB 1.7247). Text
features (keyword counts and TF-IDF) and multi-task training on the 5 non-scored
categories were all tested and gave no gain, so are not used.

Robust to the hidden re-run: data root is auto-detected, weather/image NaNs fall
back to train medians, and unseen categories/periods are handled by the encoder.

Output: <working>/submission.csv with exactly the sample_submission row_ids.
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

SCORING = ["all_drugs", "all_opioids", "all_stimulants"]
KEYS = ["period_id", "jurisdiction"]
TARGET = "rate_per_10000_ed_visits"
WEATHER = ["temp_avg_f", "precip_in"]
SIMILAR_STATES = {"AK": ["WA", "MT", "ND", "MN"], "HI": ["CA", "FL"], "DC": ["MD", "VA"]}
BG_THRESHOLD, HIGH_DENSITY = 20, 150
IMAGE_COLS = [
    "img_bg_ratio", "img_density_mean", "img_density_std", "img_density_max",
    "img_density_p90", "img_high_dens_frac", "img_top10_share", "img_spatial_spread",
]


# ----------------------------------------------------------------------------- paths
def find_data_root() -> str:
    env = os.environ.get("STAIX_DATA")
    if env and os.path.exists(f"{env}/train/dose_sys_train.csv"):
        return env
    cands = []
    if os.path.isdir("/kaggle/input"):
        cands += glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")
    cands += [".", ".."]
    for c in cands:
        if os.path.exists(f"{c}/train/dose_sys_train.csv"):
            return c
    raise FileNotFoundError("competition data root not found")


DATA = find_data_root()
WORK = os.environ.get("STAIX_WORK") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")


# --------------------------------------------------------------------------- features
def extract_one(path: str) -> dict:
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mask = np.maximum(np.maximum(r, g), b) >= BG_THRESHOLD
    n_total, n_state = mask.size, int(mask.sum())
    if n_state == 0:
        return {k: np.nan for k in IMAGE_COLS}
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    s = lum[mask]
    order = np.sort(s)
    cut = int(0.9 * n_state)
    ys, xs = np.nonzero(mask)
    cx, cy = np.average(xs, weights=s), np.average(ys, weights=s)
    spread = np.sqrt(np.average((xs - cx) ** 2 + (ys - cy) ** 2, weights=s))
    diag = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    return {
        "img_bg_ratio": 1.0 - n_state / n_total,
        "img_density_mean": float(s.mean()), "img_density_std": float(s.std()),
        "img_density_max": float(s.max()), "img_density_p90": float(np.percentile(s, 90)),
        "img_high_dens_frac": float((s > HIGH_DENSITY).mean()),
        "img_top10_share": float(order[cut:].sum() / (s.sum() + 1e-6)),
        "img_spatial_spread": float(spread / diag),
    }


def image_features(base_dir: str, cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, juris in cov[KEYS].drop_duplicates().itertuples(index=False):
        path = f"{base_dir}/images/mat_density/{juris}_{period}.png"
        feat = extract_one(path) if os.path.exists(path) else {k: np.nan for k in IMAGE_COLS}
        feat["period_id"], feat["jurisdiction"] = period, juris
        rows.append(feat)
    return pd.DataFrame(rows)


def handle_text(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["state_doh_release"] = d["state_doh_release"].fillna("")
    d["has_doh_release"] = d["state_doh_release"].str.strip().ne("").astype(int)
    return d


def impute_weather(data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Dataset C: similar-state -> train period-median -> train global-median."""
    data = data.copy()
    data["weather_missing"] = data["temp_avg_f"].isna().astype(int)
    for col in WEATHER:
        for state, donors in SIMILAR_STATES.items():
            idx = data.index[(data["jurisdiction"] == state) & (data[col].isna())]
            for i in idx:
                vals = reference.loc[
                    reference["jurisdiction"].isin(donors)
                    & (reference["period_id"] == data.at[i, "period_id"]), col].dropna()
                if not vals.empty:
                    data.at[i, col] = vals.median()
    for col in WEATHER:
        period_median = reference.groupby("period_id")[col].median()
        data[col] = data[col].fillna(data["period_id"].map(period_median)).fillna(reference[col].median())
    return data


# ------------------------------------------------------------------------- preprocess
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", make_ohe(), cat),
    ], remainder="drop")


# -------------------------------------------------------------------------------- main
def main() -> None:
    print(f"data root: {DATA}")
    target = pd.read_csv(f"{DATA}/train/dose_sys_train.csv")
    target = target[target["overdose_category"].isin(SCORING)]
    tcov = pd.read_csv(f"{DATA}/train/covariates.csv")
    vcov = pd.read_csv(f"{DATA}/val/covariates.csv")
    sub = pd.read_csv(f"{DATA}/sample_submission.csv")

    train_raw = target.merge(tcov, on=KEYS, how="left")
    val_raw = sub.merge(vcov, on=KEYS, how="left")

    print("extracting image features ...")
    train_img = image_features(f"{DATA}/train", tcov)
    val_img = image_features(f"{DATA}/val", vcov)

    train_ds = handle_text(impute_weather(train_raw, train_raw)).merge(train_img, on=KEYS, how="left")
    val_ds = handle_text(impute_weather(val_raw, train_raw)).merge(val_img, on=KEYS, how="left")

    # Robustness for the hidden window: any missing image features -> train median.
    img_median = train_ds[IMAGE_COLS].median()
    train_ds[IMAGE_COLS] = train_ds[IMAGE_COLS].fillna(img_median)
    val_ds[IMAGE_COLS] = val_ds[IMAGE_COLS].fillna(img_median)

    train_ds = train_ds.dropna(subset=[TARGET])
    drop = ["row_id", TARGET, "state_doh_release"]
    X_train = train_ds.drop(columns=drop, errors="ignore")
    y_train = train_ds[TARGET].to_numpy()
    X_val = val_ds.drop(columns=drop, errors="ignore").reindex(columns=X_train.columns)

    # Validated on 5-fold GroupKFold-by-period (notebook 14): HistGB with
    # absolute_error loss + light tuning beats the default, and a 0.7/0.3
    # HistGB/RF blend beats the 0.5/0.5 average (1.7086 vs 1.7125).
    models = {
        "HistGB": HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=600,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
    }
    weights = {"HistGB": 0.7, "RandomForest": 0.3}
    preds = []
    for name, est in models.items():
        print(f"fitting {name} ...")
        pipe = Pipeline([("pre", build_preprocessor(X_train)), ("m", est)])
        pipe.fit(X_train, y_train)
        preds.append(weights[name] * np.clip(pipe.predict(X_val), 0, None))
    ensemble = np.sum(preds, axis=0)

    out = sub[["row_id"]].copy()
    out[TARGET] = ensemble
    assert len(out) == len(sub) and out["row_id"].is_unique and out[TARGET].notna().all()

    path = f"{WORK}/submission.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(out)} rows)")
    print(out[TARGET].describe().round(3).to_string())


if __name__ == "__main__":
    main()
