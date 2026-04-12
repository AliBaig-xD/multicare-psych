"""
09_download_cnp.py
Phase 2 – Step 1: Download CNP dataset (ds000030) from OpenNeuro S3.

Dataset: UCLA Consortium for Neuropsychiatric Phenomics (CNP)
  - 290 subjects: 138 healthy, 58 schizophrenia, 49 bipolar, 45 ADHD
  - Confirmed DSM diagnoses
  - Structural T1w MRI + resting state fMRI
  - Source: OpenNeuro ds000030

We download ONLY the T1w structural scans + participants.tsv (demographics/diagnoses)
Total size: ~5-8 GB

Run:
    source env/py/bin/activate
    python scripts/09_download_cnp.py | tee logs/09_cnp_download.log
"""

import os
import subprocess
import pandas as pd

CNP_DIR       = "cnp_data"
S3_BASE       = "s3://openneuro.org/ds000030"
PARTICIPANTS  = f"{CNP_DIR}/participants.tsv"


def run(cmd: str) -> None:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    os.makedirs(CNP_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ── 1. Download participants.tsv (diagnoses + demographics) ───────────────
    print("[1/3] Downloading participants.tsv ...")
    run(f"aws s3 cp {S3_BASE}/participants.tsv {CNP_DIR}/participants.tsv --no-sign-request")

    # ── 2. Inspect participants ───────────────────────────────────────────────
    print("\n[2/3] Inspecting participants ...")
    df = pd.read_csv(PARTICIPANTS, sep="\t")
    print(f"  Total subjects : {len(df)}")
    print(f"  Columns        : {df.columns.tolist()}")
    print(f"\n  Diagnosis breakdown:")
    if "diagnosis" in df.columns:
        print(df["diagnosis"].value_counts().to_string())
    elif "group" in df.columns:
        print(df["group"].value_counts().to_string())
    else:
        print(df.head(5).to_string())

    # ── 3. Download T1w structural scans for all subjects ─────────────────────
    print("\n[3/3] Downloading T1w structural MRI scans ...")
    print("  This will download ~5-8 GB. Running in background via tmux recommended.")

    # Get subject IDs
    id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    subject_ids = df[id_col].tolist()
    print(f"  Subjects to download: {len(subject_ids)}")

    downloaded = 0
    skipped    = 0
    failed     = 0

    for sub_id in subject_ids:
        # Normalise subject ID format
        if not str(sub_id).startswith("sub-"):
            sub_id = f"sub-{sub_id}"

        out_dir  = f"{CNP_DIR}/{sub_id}/anat"
        out_file = f"{out_dir}/{sub_id}_T1w.nii.gz"

        if os.path.exists(out_file):
            skipped += 1
            continue

        os.makedirs(out_dir, exist_ok=True)

        s3_path = f"{S3_BASE}/{sub_id}/anat/{sub_id}_T1w.nii.gz"
        cmd = f"aws s3 cp {s3_path} {out_file} --no-sign-request --quiet"

        try:
            result = subprocess.run(cmd, shell=True, timeout=120)
            if result.returncode == 0:
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"  Downloaded {downloaded}/{len(subject_ids)} ...")
            else:
                failed += 1
        except subprocess.TimeoutExpired:
            print(f"  Timeout: {sub_id}")
            failed += 1

    print(f"\nDownload complete:")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped} (already exist)")
    print(f"  Failed     : {failed}")
    print(f"\nNext step: run scripts/10_preprocess_cnp.py")


if __name__ == "__main__":
    main()