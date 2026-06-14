"""
STAI-X Challenge 2026 — Step 8: per-model gain from the image features.

For EVERY model (Model 0-4) on EVERY weather dataset (A/B/C), measure how much
the 8 MAT-density image features (from 06) improve performance, using the same
5-fold GroupKFold-by-period engine as notebooks 04/05/07 so the numbers line up.

The image features carry no per-fold statistics (each is computed from one image
independently in 06), so merging them onto the raw table before CV is leak-free.

Writes outputs/image_feature_gains.csv and prints a Markdown summary table.

Run (from repo root):
    python notebooks/08_image_feature_gains.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from statx_helpers import DATASET_SPECS, cross_val_by_period

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
KEYS = ["period_id", "jurisdiction"]
N_SPLITS = 5


def main() -> None:
    raw = pd.read_csv(OUT / "train_universal_merged.csv")
    feats = pd.read_csv(OUT / "image_features.csv")
    feats = feats[feats["split"] == "train"].drop(columns="split")
    raw_img = raw.merge(feats, on=KEYS, how="left")

    rows = []
    for dataset_name, spec in DATASET_SPECS.items():
        method = spec["method"]
        cov = {r["model"]: r for r in cross_val_by_period(raw, method, dataset_name, n_splits=N_SPLITS)}
        img = {r["model"]: r for r in cross_val_by_period(raw_img, method, dataset_name, n_splits=N_SPLITS)}
        for model in cov:
            c, i = cov[model], img[model]
            rows.append({
                "dataset": dataset_name,
                "model": model,
                "RMSE_cov": c["RMSE"], "RMSE_img": i["RMSE"],
                "RMSE_impr_%": (c["RMSE"] - i["RMSE"]) / c["RMSE"] * 100,
                "MAE_cov": c["MAE"], "MAE_img": i["MAE"],
                "MAE_impr_%": (c["MAE"] - i["MAE"]) / c["MAE"] * 100,
                "R2_cov": c["R2"], "R2_img": i["R2"], "R2_delta": i["R2"] - c["R2"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "image_feature_gains.csv", index=False)
    print(f"Saved: outputs/image_feature_gains.csv ({len(df)} rows)\n")

    # Markdown table (MAE-focused) ready to paste into the README.
    print("| Dataset | Model | MAE (cov) | MAE (+img) | MAE improve | RMSE improve |")
    print("|---|---|---|---|---|---|")
    short = {
        "Model 0 - Mean baseline": "Model 0 baseline",
        "Model 1 - Ridge fixed effects": "Model 1 Ridge",
        "Model 2 - ElasticNet fixed effects": "Model 2 ElasticNet",
        "Model 3 - HistGradientBoosting": "Model 3 HistGB",
        "Model 4 - RandomForest": "Model 4 RandomForest",
    }
    order = list(short)
    for ds in DATASET_SPECS:
        sub = df[df["dataset"] == ds].set_index("model")
        for m in order:
            r = sub.loc[m]
            print(f"| {ds} | {short[m]} | {r['MAE_cov']:.3f} | {r['MAE_img']:.3f} "
                  f"| {r['MAE_impr_%']:+.1f}% | {r['RMSE_impr_%']:+.1f}% |")


if __name__ == "__main__":
    main()
