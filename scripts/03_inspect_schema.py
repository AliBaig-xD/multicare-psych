"""
03_inspect_schema.py
Phase A – Step 3: Inspect the psych_brain_multimodal directory and print
everything we need to know before writing the master table script.

This is a READ-ONLY diagnostic — it creates no output files.
Run it after 02_build_psych_subset.py and BEFORE 04_build_master_table.py.

Run:
    source env/py/bin/activate
    python scripts/03_inspect_schema.py | tee logs/03_schema.log
"""

import json
import os

import pandas as pd

BASE_DIR = "medical_datasets/psych_brain_multimodal"


def inspect_dir(base: str) -> None:
    print(f"\n{'='*60}")
    print(f"Directory listing: {base}")
    print("="*60)
    for root, dirs, files in os.walk(base):
        # Skip image-heavy sub-dirs to keep output readable
        depth = root.replace(base, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth < 2:
            for f in files[:20]:          # show first 20 files per folder
                print(f"{indent}  {f}")
            if len(files) > 20:
                print(f"{indent}  ... ({len(files) - 20} more files)")


def inspect_metadata(base: str) -> None:
    candidates = [
        f for f in os.listdir(base)
        if f.endswith(".json") or f.endswith(".csv") or f.endswith(".parquet")
    ]
    if not candidates:
        print("\nWARNING: No metadata file (.json / .csv / .parquet) found at top level.")
        print("Check sub-directories above and update 04_build_master_table.py accordingly.")
        return

    print(f"\nMetadata files found: {candidates}")

    for fname in candidates:
        path = os.path.join(base, fname)
        print(f"\n--- {fname} ---")
        try:
            if fname.endswith(".json"):
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    print("  JSON is not a list — printing top-level keys:", list(data.keys()))
                    continue
            elif fname.endswith(".csv"):
                df = pd.read_csv(path, nrows=5)
            else:
                df = pd.read_parquet(path)

            print(f"  Shape  : {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print("\n  First 3 rows:")
            print(df.head(3).to_string())

            print("\n  Null counts:")
            print(df.isnull().sum().to_string())

        except Exception as e:
            print(f"  Could not parse {fname}: {e}")


def main():
    if not os.path.isdir(BASE_DIR):
        print(f"ERROR: {BASE_DIR} does not exist.")
        print("Run scripts/02_build_psych_subset.py first.")
        raise SystemExit(1)

    inspect_dir(BASE_DIR)
    inspect_metadata(BASE_DIR)

    print("\n" + "="*60)
    print("ACTION REQUIRED:")
    print("  Review the column names above, then open")
    print("  scripts/04_build_master_table.py and update the")
    print("  COLUMN MAPPING section at the top of the file.")
    print("="*60)


if __name__ == "__main__":
    main()
