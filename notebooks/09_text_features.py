"""
STAI-X Challenge 2026 — Step 9: DOH press-release text features.

A cleaned, reproducible rewrite of the old STAIX-26.py exploration. Turns the
`state_doh_release` free text into interpretable keyword-count features, keyed on
(period_id, jurisdiction) so they merge directly onto the modeling tables (same
pattern as the image features in 06).

What it produces per (split, period_id, jurisdiction):
  - txt_crisis / txt_alert / txt_action : keyword-group counts
  - txt_len        : number of word tokens (release volume proxy)
  - txt_has_release: 1 if any qualifying release text is present
  - txt_risk_class : argmax group label (CRISIS / ALERT / ACTION / UNKNOWN)

Each feature is computed from ONE row's text independently — no cross-row or
cross-fold statistics — so merging before cross-validation is leak-free.

Fixes vs the original STAIX-26.py:
  - relative paths (runs from the repo) instead of hardcoded /Users/... paths;
  - ONE shared keyword list for train and val (the original used different alert
    words per split, making the feature inconsistent);
  - the validation labels are computed from the validation frame (the original
    bug assigned train's classification to val);
  - de-duplicated keyword lists, dead `adjustment` column removed;
  - writes outputs/text_features.csv (the original wrote nothing).

Run (from repo root):
    python notebooks/09_text_features.py
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "train"
VAL_DIR = ROOT / "val"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

KEYS = ["period_id", "jurisdiction"]

# One shared keyword set for BOTH splits (the original diverged between them).
KEYWORD_GROUPS = {
    "txt_crisis": ["overdose", "fentanyl", "xylazine", "opioid"],
    "txt_alert": ["suspected", "reported", "cluster", "detected", "emergency",
                  "preliminary", "witnessing", "surveillance", "wounds"],
    "txt_action": ["treatment", "medication", "programs", "recovery", "services",
                   "naloxone", "enforcement", "reduction"],
}
# Routine surveillance-reporting language: down-weight crisis by 1 (keeps the
# original author's intent, applied identically to both splits).
ROUTINE_RE = r"reported|data|surveillance"
WORD_RE = r"\b[a-z]{3,}\b"
SCORE_COLS = ["txt_crisis", "txt_alert", "txt_action"]
FEATURE_COLS = SCORE_COLS + ["txt_len", "txt_has_release"]


def classify(row) -> str:
    """argmax keyword group; UNKNOWN when no group fired."""
    scores = {"CRISIS": row["txt_crisis"], "ALERT": row["txt_alert"], "ACTION": row["txt_action"]}
    top = max(scores, key=scores.get)
    return top if scores[top] > 0 else "UNKNOWN"


def extract(cov: pd.DataFrame) -> pd.DataFrame:
    out = cov[KEYS].drop_duplicates().reset_index(drop=True)
    text = cov["state_doh_release"].fillna("").str.lower()

    for col, words in KEYWORD_GROUPS.items():
        out[col] = text.str.count("|".join(words)).to_numpy()
    out["txt_crisis"] -= text.str.contains(ROUTINE_RE, na=False).astype(int).to_numpy()

    out["txt_len"] = text.apply(lambda t: len(re.findall(WORD_RE, t))).to_numpy()
    out["txt_has_release"] = text.str.strip().ne("").astype(int).to_numpy()
    out["txt_risk_class"] = out.apply(classify, axis=1)
    return out


def main() -> None:
    frames = []
    for split, base in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        cov = pd.read_csv(base / "covariates.csv")
        feats = extract(cov)
        feats.insert(0, "split", split)
        frames.append(feats)
        print(f"{split}: {len(feats)} rows | "
              f"with release: {int(feats['txt_has_release'].sum())} | "
              + feats["txt_risk_class"].value_counts().to_dict().__str__())

    out = pd.concat(frames, ignore_index=True)
    cols = ["split"] + KEYS + FEATURE_COLS + ["txt_risk_class"]
    out[cols].to_csv(OUT_DIR / "text_features.csv", index=False)
    print(f"\nWrote outputs/text_features.csv ({len(out)} rows, {len(FEATURE_COLS)} numeric features)")


if __name__ == "__main__":
    main()
