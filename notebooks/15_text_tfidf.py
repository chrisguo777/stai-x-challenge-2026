"""
STAI-X Challenge 2026 — Step 15: TF-IDF text features (leak-free).

Keyword counts (09/10) gave no gain. TF-IDF is the next step, aimed at the
hardest category all_drugs. To avoid leakage the vectorizer is fit INSIDE each
fold (a sklearn step, not a precomputed CSV): the model Pipeline contains a
ColumnTransformer with TfidfVectorizer(state_doh_release) -> TruncatedSVD, so it
re-fits on each fold's training text only.

Compares, on the same 5-fold GroupKFold-by-period (scored-only MAE), HistGB
(absolute_error) with vs without the TF-IDF block. Set MULTITASK to match the
notebook 13 verdict.

Run (from repo root):
    python notebooks/15_text_tfidf.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from statx_helpers import make_dataset, make_ohe, TARGET_COL, PERIOD_COL

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
KEYS = ["period_id", "jurisdiction"]
METHOD = "similar_state"
N_SPLITS = 5
SCORED = ["all_drugs", "all_opioids", "all_stimulants"]
MULTITASK = False                 # notebook 13 verdict: multitask hurt, use scored-only
TEXT_COL = "state_doh_release"
IMAGE_COLS = [
    "img_bg_ratio", "img_density_mean", "img_density_std", "img_density_max",
    "img_density_p90", "img_high_dens_frac", "img_top10_share", "img_spatial_spread",
]


def load_raw() -> pd.DataFrame:
    target = pd.read_csv(ROOT / "train" / "dose_sys_train.csv")
    if not MULTITASK:
        target = target[target["overdose_category"].isin(SCORED)]
    cov = pd.read_csv(ROOT / "train" / "covariates.csv")
    img = pd.read_csv(OUT / "image_features.csv")
    img = img[img["split"] == "train"][KEYS + IMAGE_COLS]
    return (target.merge(cov, on=KEYS, how="left").merge(img, on=KEYS, how="left")
            .dropna(subset=[TARGET_COL]).reset_index(drop=True))


def build_pre(X: pd.DataFrame, with_tfidf: bool) -> ColumnTransformer:
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = [c for c in X.select_dtypes(include=["object"]).columns if c != TEXT_COL]
    transformers = [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", make_ohe(), cat),
    ]
    if with_tfidf:
        text = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=2000, min_df=3, ngram_range=(1, 2),
                                      stop_words="english")),
            ("svd", TruncatedSVD(n_components=50, random_state=42)),
        ])
        transformers.append(("txt", text, TEXT_COL))     # scalar -> 1D Series for the vectorizer
    return ColumnTransformer(transformers, remainder="drop")


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def main() -> None:
    raw = load_raw()
    print(f"MULTITASK={MULTITASK} | rows={len(raw)} | model=HistGB(absolute_error)\n")
    gkf = GroupKFold(n_splits=N_SPLITS)
    periods = raw[PERIOD_COL].to_numpy()

    res = {"no_tfidf": [], "tfidf": []}
    per_cat = {("no_tfidf", c): [] for c in SCORED}
    per_cat.update({("tfidf", c): [] for c in SCORED})

    for tr_idx, te_idx in gkf.split(raw, raw[TARGET_COL], periods):
        train_ds = make_dataset(raw.iloc[tr_idx], METHOD)
        test_raw = raw.iloc[te_idx]
        test_raw = test_raw[test_raw["overdose_category"].isin(SCORED)]
        test_ds = make_dataset(test_raw, METHOD, reference=raw.iloc[tr_idx])

        drop = ["row_id", TARGET_COL]                    # keep TEXT_COL for tfidf
        X_tr = train_ds.drop(columns=drop, errors="ignore")
        y_tr = train_ds[TARGET_COL].to_numpy()
        X_te = test_ds.drop(columns=drop, errors="ignore").reindex(columns=X_tr.columns)
        y_te = test_ds[TARGET_COL].to_numpy()
        cats = test_ds["overdose_category"].to_numpy()

        for cond, with_tfidf in [("no_tfidf", False), ("tfidf", True)]:
            pipe = Pipeline([
                ("pre", build_pre(X_tr, with_tfidf)),
                ("m", HistGradientBoostingRegressor(loss="absolute_error", random_state=42)),
            ])
            pipe.fit(X_tr, y_tr)
            p = np.clip(pipe.predict(X_te), 0, None)
            res[cond].append(mae(y_te, p))
            for c in SCORED:
                m = cats == c
                per_cat[(cond, c)].append(mae(y_te[m], p[m]))

    print("  5-fold scored MAE (HistGB absolute_error):")
    for cond in ["no_tfidf", "tfidf"]:
        print(f"    {cond:<10}{np.mean(res[cond]):.4f}")
    print("\n  per-category MAE:")
    print(f"    {'cond':<10}" + "".join(f"{c:>16}" for c in SCORED))
    for cond in ["no_tfidf", "tfidf"]:
        print(f"    {cond:<10}" + "".join(f"{np.mean(per_cat[(cond,c)]):>16.4f}" for c in SCORED))


if __name__ == "__main__":
    main()
