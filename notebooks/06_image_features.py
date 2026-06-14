"""
STAI-X Challenge 2026 — Step 6: MAT-density image features.

Produces interpretable features from the per-(jurisdiction, period) MAT-density
heatmaps, keyed on (period_id, jurisdiction) so they merge directly onto the
modeling tables from notebooks 01-03 (train_*_merged.csv / *_period_*.csv).

Pipeline:
  1. Build the UNIQUE (split, period_id, jurisdiction) work-list from the raw
     covariates. Each (period, state) has exactly one image, identical across
     the 3 scoring categories, so we process each image ONCE (not 3x).
  2. For each viridis heatmap, mask out the black out-of-state background and
     compute 8 interpretable density statistics over the in-state region.
  3. Write outputs/image_features.csv.

Why these features and not a CNN: the per-feature signal is weak (|Spearman rho|
<= 0.22 vs target) and partly state-bound, so a CNN would mostly overfit. Under
the project's period-split validation, adding these features lowers MAE on the
3 'all_*' targets by ~7% (1.85 -> 1.73) — a modest, robust gain. See
07_validate_image_features.py.

Run (from repo root):
    python notebooks/06_image_features.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent      # repo root (notebooks/ -> root)
TRAIN_DIR = ROOT / "train"
VAL_DIR = ROOT / "val"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

KEYS = ["period_id", "jurisdiction"]
BG_THRESHOLD = 20       # max(R,G,B) below this -> outside-state black background
HIGH_DENSITY = 150      # luminance above this -> "high MAT density" pixel

FEATURE_COLS = [
    "img_bg_ratio", "img_density_mean", "img_density_std", "img_density_max",
    "img_density_p90", "img_high_dens_frac", "img_top10_share", "img_spatial_spread",
]


def build_manifest() -> pd.DataFrame:
    """One row per UNIQUE (split, period_id, jurisdiction): image path."""
    rows = []
    for split, base in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        cov = pd.read_csv(base / "covariates.csv")
        m = cov[KEYS].drop_duplicates().copy()
        m.insert(0, "split", split)
        m["mat_density_image"] = (
            f"{split}/images/mat_density/"
            + m["jurisdiction"] + "_" + m["period_id"] + ".png"
        )
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def extract_one(rel_path: str) -> dict:
    """8 interpretable density features from one heatmap PNG.

    viridis: dark purple = LOW density, bright yellow = HIGH. Luminance is
    monotone in the scale, so it is a clean density proxy. Pure-black pixels
    are outside the state polygon and are masked out so state size / background
    area does not pollute the features.
    """
    img = np.asarray(Image.open(ROOT / rel_path).convert("RGB"), dtype=np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    mask = maxc >= BG_THRESHOLD                  # True = inside state

    n_total = mask.size
    n_state = int(mask.sum())
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b   # density proxy
    state_lum = lum[mask]

    if n_state == 0:
        return {k: np.nan for k in FEATURE_COLS}

    order = np.sort(state_lum)
    top10_cut = int(0.9 * n_state)
    total_density = state_lum.sum() + 1e-6

    # Spatial dispersion: luminance-weighted std of pixel positions / diagonal.
    # High = clinics spread across the state; low = one concentrated hotspot.
    ys, xs = np.nonzero(mask)
    w = state_lum
    cx = np.average(xs, weights=w)
    cy = np.average(ys, weights=w)
    spread = np.sqrt(np.average((xs - cx) ** 2 + (ys - cy) ** 2, weights=w))
    diag = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)

    return {
        "img_bg_ratio":       1.0 - n_state / n_total,
        "img_density_mean":   float(state_lum.mean()),
        "img_density_std":    float(state_lum.std()),
        "img_density_max":    float(state_lum.max()),
        "img_density_p90":    float(np.percentile(state_lum, 90)),
        "img_high_dens_frac": float((state_lum > HIGH_DENSITY).mean()),
        "img_top10_share":    float(order[top10_cut:].sum() / total_density),
        "img_spatial_spread": float(spread / diag),
    }


def main() -> None:
    manifest = build_manifest()
    print(f"Extracting features for {len(manifest)} unique images ...")

    feats = []
    for i, row in enumerate(manifest.itertuples(index=False)):
        f = extract_one(row.mat_density_image)
        f["split"] = row.split
        f["period_id"] = row.period_id
        f["jurisdiction"] = row.jurisdiction
        feats.append(f)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(manifest)}")

    out = pd.DataFrame(feats)[["split"] + KEYS + FEATURE_COLS]
    out_path = OUT_DIR / "image_features.csv"
    out.to_csv(out_path, index=False)

    print(f"\nWrote outputs/image_features.csv  ({len(out)} rows, {len(FEATURE_COLS)} features)")
    print(out["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
