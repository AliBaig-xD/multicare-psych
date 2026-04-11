"""
04_build_master_table.py
Phase B: Build a unified master table — one row per (case, image) pair.

Strategy (confirmed from schema inspection):
  1. Load psych cases from psych_brain_multimodal/cases.csv
  2. Load full image metadata from whole_multicare_dataset/captions_and_labels.csv
  3. Join on PMCID, filter to brain/head radiology only
  4. Resolve exact file paths using PMC folder structure
  5. Save to data/psych_brain_master.parquet

Output:
  data/psych_brain_master.parquet  (1,515 rows expected)

Run:
    source env/py/bin/activate
    python scripts/04_build_master_table.py | tee logs/04_master_table.log
"""

import os
import pandas as pd

PSYCH_CASES_CSV  = "medical_datasets/psych_brain_multimodal/cases.csv"
CAPTIONS_CSV     = "medical_datasets/whole_multicare_dataset/captions_and_labels.csv"
WHOLE_BASE       = "medical_datasets/whole_multicare_dataset"
OUT_PATH         = "data/psych_brain_master.parquet"

BRAIN_REGIONS    = ["head", "brain", "spine"]
PMC_DIRS         = ["PMC1","PMC2","PMC3","PMC4","PMC5","PMC6","PMC7","PMC8","PMC9"]


def resolve_image_path(fname: str, pmcid: str) -> str | None:
    """Find the actual file path for an image given its filename and PMCID."""
    short = pmcid[:5]  # e.g. PMC10
    for pmc_dir in PMC_DIRS:
        candidate = os.path.join(WHOLE_BASE, pmc_dir, short, fname)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    os.makedirs("data", exist_ok=True)

    # 1. Load psych cases
    cases = pd.read_csv(PSYCH_CASES_CSV)
    print(f"Psych cases        : {len(cases)}")
    psych_pmcids = set(cases['case_id'].str.replace(r'_\d+$', '', regex=True))
    print(f"Unique PMCIDs      : {len(psych_pmcids)}")

    # 2. Load full image metadata
    captions = pd.read_csv(CAPTIONS_CSV)
    print(f"Full image metadata: {len(captions)}")

    # 3. Join: match captions to psych cases via PMCID
    captions['pmcid'] = captions['patient_id'].str.replace(r'_\d+$', '', regex=True)
    matched = captions[captions['pmcid'].isin(psych_pmcids)].copy()
    print(f"Matched to psych   : {len(matched)}")

    # 4. Filter to brain/head radiology only
    brain = matched[matched['radiology_region'].isin(BRAIN_REGIONS)].copy()
    print(f"Brain/head radiology: {len(brain)}")

    # 5. Merge case text in
    # cases.csv case_id is like PMC10064861_01, patient_id in captions is also PMC10064861_01
    brain = brain.merge(
        cases[['case_id', 'case_text', 'gender', 'age']],
        left_on='patient_id',
        right_on='case_id',
        how='left'
    )

    # 6. Resolve actual image paths
    print("Resolving image paths...")
    brain['image_path'] = brain.apply(
        lambda r: resolve_image_path(r['file'], r['pmcid']), axis=1
    )

    # 7. Drop any unresolved
    before = len(brain)
    brain = brain.dropna(subset=['image_path', 'case_text'])
    print(f"Dropped {before - len(brain)} rows (unresolved path or missing text)")

    # 8. Final clean table
    master = brain[[
        'patient_id',
        'case_text',
        'image_path',
        'caption',
        'image_type',
        'image_subtype',
        'radiology_region',
        'radiology_view',
        'ml_labels_for_supervised_classification',
        'gender',
        'age',
        'pmcid',
    ]].copy()

    master = master.rename(columns={
        'patient_id':                              'case_id',
        'ml_labels_for_supervised_classification': 'image_labels',
    })

    master = master.reset_index(drop=True)
    master.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(master)} rows to {OUT_PATH}")
    print("\nSample:")
    print(master[['case_id', 'image_path', 'image_type', 'radiology_region']].head(5).to_string())
    print("\nNext step: run scripts/05_embed_cases.py")


if __name__ == "__main__":
    main()