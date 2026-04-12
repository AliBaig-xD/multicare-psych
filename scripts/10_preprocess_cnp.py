"""
10_preprocess_cnp.py
Phase 2 – Step 2: Preprocess CNP NIfTI files.

For each T1w MRI scan:
  1. Load the NIfTI volume
  2. Reorient to standard orientation (RAS)
  3. Extract 3 canonical 2D slices (axial, sagittal, coronal) at brain centre
  4. Save as PNG for CLIP embedding
  5. Also save volume metadata for 3D embedding

Output:
  cnp_data/slices/         PNG slices for CLIP
  cnp_data/cnp_master.parquet   metadata table

Run:
    source env/py/bin/activate
    python scripts/10_preprocess_cnp.py | tee logs/10_preprocess.log
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import reorder_img
from PIL import Image
from tqdm import tqdm

CNP_DIR       = "cnp_data"
SLICES_DIR    = f"{CNP_DIR}/slices"
PARTICIPANTS  = f"{CNP_DIR}/participants.tsv"
OUT_PARQUET   = f"{CNP_DIR}/cnp_master.parquet"


def normalise_slice(arr: np.ndarray) -> np.ndarray:
    """Normalise a 2D slice to 0-255 uint8."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn) * 255
    return arr.astype(np.uint8)


def extract_slices(nii_path: str, sub_id: str) -> list[dict]:
    """Extract axial, sagittal, coronal slices from a NIfTI volume."""
    img    = nib.load(nii_path)
    img    = reorder_img(img, resample="continuous")   # reorient to RAS
    data   = img.get_fdata()

    cx, cy, cz = [s // 2 for s in data.shape[:3]]

    slices = {
        "axial":    data[:, :, cz],
        "sagittal": data[cx, :, :],
        "coronal":  data[:, cy, :],
    }

    results = []
    for view, arr in slices.items():
        arr   = normalise_slice(arr)
        # Rotate so brain is upright
        arr   = np.rot90(arr)
        img2d = Image.fromarray(arr).convert("RGB")
        # Resize to 224x224 for CLIP
        img2d = img2d.resize((224, 224), Image.LANCZOS)

        out_path = os.path.join(SLICES_DIR, f"{sub_id}_{view}.png")
        img2d.save(out_path)

        results.append({
            "sub_id":     sub_id,
            "nii_path":   nii_path,
            "slice_path": out_path,
            "view":       view,
        })

    return results


def main():
    os.makedirs(SLICES_DIR, exist_ok=True)

    # Load participant metadata
    participants = pd.read_csv(PARTICIPANTS, sep="\t")
    id_col   = "participant_id" if "participant_id" in participants.columns else participants.columns[0]
    diag_col = "diagnosis" if "diagnosis" in participants.columns else "group"

    print(f"Participants: {len(participants)}")
    print(f"Diagnosis column: {diag_col}")
    print(participants[diag_col].value_counts().to_string())

    rows = []
    missing = 0

    for _, row in tqdm(participants.iterrows(), total=len(participants), desc="Processing"):
        sub_id = str(row[id_col])
        if not sub_id.startswith("sub-"):
            sub_id = f"sub-{sub_id}"

        nii_path = f"{CNP_DIR}/{sub_id}/anat/{sub_id}_T1w.nii.gz"

        if not os.path.exists(nii_path):
            missing += 1
            continue

        try:
            slices = extract_slices(nii_path, sub_id)
            for s in slices:
                s["diagnosis"] = str(row.get(diag_col, "unknown"))
                s["age"]       = row.get("age", None)
                s["gender"]    = row.get("gender", row.get("sex", None))
            rows.extend(slices)
        except Exception as e:
            print(f"  Error processing {sub_id}: {e}")
            missing += 1

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"\nProcessed    : {len(df) // 3} subjects")
    print(f"Total slices : {len(df)}")
    print(f"Missing/error: {missing}")
    print(f"\nDiagnosis breakdown in processed data:")
    print(df.drop_duplicates("sub_id")["diagnosis"].value_counts().to_string())
    print(f"\nSaved to {OUT_PARQUET}")
    print("Next step: run scripts/11_embed_cnp.py")


if __name__ == "__main__":
    main()
