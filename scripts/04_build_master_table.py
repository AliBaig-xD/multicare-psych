"""
04_build_master_table.py
Phase B: Build a unified master table — one row per (case, image) pair.

Schema confirmed from actual psych_brain_multimodal output:
  cases.csv          → case_id, pmcid, gender, age, case_text
  image_metadata.json → file_id, file, file_path, main_image, case_id,
                        license, file_size, caption, image_type,
                        image_subtype, radiology_region, radiology_view,
                        ml_labels_for_supervised_classification, ...

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
OUT_PATH = "data/psych_brain_master.parquet"


def load_cases() -> pd.DataFrame:
    path = os.path.join(BASE_DIR, "cases.csv")
    df = pd.read_csv(path)
    print(f"cases.csv          : {df.shape[0]} rows, columns: {df.columns.tolist()}")
    return df


def load_image_metadata() -> pd.DataFrame:
    path = os.path.join(BASE_DIR, "image_metadata.json")
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(lines)
    print(f"image_metadata.json: {df.shape[0]} rows, columns: {df.columns.tolist()}")
    return df


def main():
    os.makedirs("data", exist_ok=True)

    cases  = load_cases()
    images = load_image_metadata()

    # Join on case_id
    master = images.merge(cases, on="case_id", how="inner")
    print(f"\nAfter join         : {len(master)} rows")

    # Keep and rename the columns we need downstream
    master = master[[
        "case_id",
        "case_text",
        "file_path",        # full relative path to the image file
        "caption",
        "image_type",
        "image_subtype",
        "radiology_region",
        "radiology_view",
        "ml_labels_for_supervised_classification",
        "gender",
        "age",
        "pmcid",
    ]].copy()

    master = master.rename(columns={
        "file_path":                               "image_path",
        "ml_labels_for_supervised_classification": "image_labels",
    })

    # Drop rows missing image path or case text
    before = len(master)
    master = master.dropna(subset=["image_path", "case_text"])
    print(f"Dropped {before - len(master)} rows with missing image_path or case_text")

    master = master.reset_index(drop=True)

    master.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(master)} rows to {OUT_PATH}")
    print("\nSample:")
    print(master[["case_id", "image_path", "image_type", "radiology_region"]].head(5).to_string())
    print("\nNext step: run scripts/05_embed_cases.py")


if __name__ == "__main__":
    main()