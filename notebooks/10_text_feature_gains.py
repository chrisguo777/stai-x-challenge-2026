"""
STAI-X Challenge 2026 — Step 10: gain from the DOH text features.

Measures how much the text features (from 09) move the needle on the final
setup (Universal + Dataset C), for every model, on the same 5-fold
GroupKFold-by-period engine as 04/05/07/08. Reports four feature sets so the
contribution of text is visible both alone and on top of the image features:

    cov              : team covariates only
    cov + text       : + DOH text features (09)
    cov + image      : + MAT-density image features (06)
    cov + image+text : the full multimodal stack

Both text and image features are per-row independent (no cross-fold statistics),
so merging them onto the raw table before CV is leak-free.

Writes outputs/text_feature_gains.csv and prints a Markdown table.

Run (from repo root):
    python notebooks/10_text_feature_gains.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from statx_helpers import cross_val_by_period

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
KEYS = ["period_id", "jurisdiction"]
METHOD, DATASET = "similar_state", "Dataset C"   # the chosen weather variant
N_SPLITS = 5

IMAGE_COLS = [
    "img_bg_ratio", "img_density_mean", "img_density_std", "img_density_max",
    "img_density_p90", "img_high_dens_frac", "img_top10_share", "img_spatial_spread",
]
TEXT_COLS = ["txt_crisis", "txt_alert", "txt_action", "txt_len", "txt_has_release"]

SHORT = {
    "Model 0 - Mean baseline": "Model 0 baseline",
    "Model 1 - Ridge fixed effects": "Model 1 Ridge",
    "Model 2 - ElasticNet fixed effects": "Model 2 ElasticNet",
    "Model 3 - HistGradientBoosting": "Model 3 HistGB",
    "Model 4 - RandomForest": "Model 4 RandomForest",
}


def train_features(name: str, cols: list) -> pd.DataFrame:
    """Train-folder rows of a feature CSV, numeric cols only."""
    f = pd.read_csv(OUT / name)
    return f[f["split"] == "train"][KEYS + cols]


def scored(raw: pd.DataFrame) -> dict:
    rows = cross_val_by_period(raw, METHOD, DATASET, n_splits=N_SPLITS)
    return {r["model"]: r for r in rows}


def main() -> None:
    raw = pd.read_csv(OUT / "train_universal_merged.csv")
    img = train_features("image_features.csv", IMAGE_COLS)
    txt = train_features("text_features.csv", TEXT_COLS)

    raw_txt = raw.merge(txt, on=KEYS, how="left")
    raw_img = raw.merge(img, on=KEYS, how="left")
    raw_both = raw_img.merge(txt, on=KEYS, how="left")

    sets = {
        "cov": scored(raw),
        "cov+text": scored(raw_txt),
        "cov+image": scored(raw_img),
        "cov+image+text": scored(raw_both),
    }

    rows = []
    for model in sets["cov"]:
        rec = {"dataset": DATASET, "model": model}
        for name, res in sets.items():
            rec[f"MAE_{name}"] = res[model]["MAE"]
            rec[f"RMSE_{name}"] = res[model]["RMSE"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "text_feature_gains.csv", index=False)
    print(f"Saved: outputs/text_feature_gains.csv ({len(df)} rows)\n")

    base, full = "MAE_cov", "MAE_cov+image+text"
    print(f"Dataset C, 5-fold, universal — MAE by feature set")
    print("| Model | cov | +text | +image | +image+text | text vs cov | full vs cov |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        t_impr = (r["MAE_cov"] - r["MAE_cov+text"]) / r["MAE_cov"] * 100 if r["MAE_cov"] else 0
        f_impr = (r[base] - r[full]) / r[base] * 100 if r[base] else 0
        print(f"| {SHORT[r['model']]} | {r['MAE_cov']:.3f} | {r['MAE_cov+text']:.3f} "
              f"| {r['MAE_cov+image']:.3f} | {r['MAE_cov+image+text']:.3f} "
              f"| {t_impr:+.1f}% | {f_impr:+.1f}% |")


if __name__ == "__main__":
    main()
