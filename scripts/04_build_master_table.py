"""
04_build_master_table.py
Phase B: Build a unified master table — one row per (case, image) pair.

BEFORE RUNNING:
  1. Run scripts/03_inspect_schema.py
  2. Look at the printed column names
  3. Update the COLUMN MAPPING section below to match your actual schema

Output:
  data/psych_brain_master.parquet

Run:
    source env/py/bin/activate
    python scripts/04_build_master_table.py | tee logs/04_master_table.log
"""

import json
import os

import pandas as pd

BASE_DIR = "medical_datasets/psych_brain_multimodal"
OUT_PARQUET = "data/psych_brain_master.parquet"

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN MAPPING
# Update these values after running 03_inspect_schema.py.
# Set each to the actual column name you see in the metadata file.
# ──────────────────────────────────────────────────────────────────────────────
CASE_ID_COL    = "case_id"       # unique identifier for the case
TEXT_COL       = "case_text"     # full narrative text of the case
IMAGE_PATH_COL = "image_path"    # path to the image (relative to BASE_DIR)
CAPTION_COL    = "caption"       # image caption
LABEL_COL      = "label"         # image label(s)
# ──────────────────────────────────────────────────────────────────────────────


def load_raw(base: str) -> pd.DataFrame:
    """Load the metadata file from the subset directory."""
    candidates = [
        f for f in os.listdir(base)
        if f.endswith(".json") or f.endswith(".csv") or f.endswith(".parquet")
    ]
    if not candidates:
        raise RuntimeError(
            f"No metadata file found in {base}. "
            "Run 03_inspect_schema.py to investigate."
        )

    path = os.path.join(base, candidates[0])
    print(f"Loading metadata from: {path}")

    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        return pd.read_parquet(path)


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "CASE_ID_COL":    CASE_ID_COL,
        "TEXT_COL":       TEXT_COL,
        "IMAGE_PATH_COL": IMAGE_PATH_COL,
        "CAPTION_COL":    CAPTION_COL,
        "LABEL_COL":      LABEL_COL,
    }
    missing = [
        f"{var}={col!r}" for var, col in required.items()
        if col not in df.columns
    ]
    if missing:
        raise RuntimeError(
            "Column mapping mismatch. Update the COLUMN MAPPING section.\n"
            "Missing columns: " + ", ".join(missing) + "\n"
            f"Available columns: {df.columns.tolist()}"
        )


def main():
    os.makedirs("data", exist_ok=True)

    df = load_raw(BASE_DIR)
    print(f"Raw metadata shape: {df.shape}")

    validate_columns(df)

    master = df[[
        CASE_ID_COL,
        TEXT_COL,
        IMAGE_PATH_COL,
        CAPTION_COL,
        LABEL_COL,
    ]].copy()

    master = master.rename(columns={
        CASE_ID_COL:    "case_id",
        TEXT_COL:       "case_text",
        IMAGE_PATH_COL: "image_path",
        CAPTION_COL:    "image_caption",
        LABEL_COL:      "image_labels",
    })

    # Drop rows with no image path or no text
    before = len(master)
    master = master.dropna(subset=["image_path", "case_text"])
    print(f"Dropped {before - len(master)} rows with missing image_path or case_text.")

    # Reset index so iloc lookups work cleanly downstream
    master = master.reset_index(drop=True)

    master.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved {len(master)} rows to {OUT_PARQUET}")
    print("\nSample:")
    print(master.head(3).to_string())
    print("\nNext step: run scripts/05_embed_cases.py")


if __name__ == "__main__":
    main()
