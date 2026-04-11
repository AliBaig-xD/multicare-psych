"""
02_build_psych_subset.py
Phase A – Step 2: Filter MultiCaRe down to brain-imaging psychiatric cases.

Filters applied:
  - Adult cases (min_age >= 18)
  - Case text OR image caption mentions a psychiatric term
  - Image label suggests brain / head imaging

Output:
  ./medical_datasets/psych_brain_multimodal/   (images + metadata)

Run:
    source env/py/bin/activate
    python scripts/02_build_psych_subset.py | tee logs/02_psych_subset.log
"""

from multiversity.multicare_dataset import MedicalDatasetCreator


PSYCH_TERMS = [
    "schizophrenia", "bipolar", "mania", "manic",
    "depression", "major depressive", "mdd",
    "psychosis", "psychotic",
    "anxiety", "panic disorder",
    "ocd", "obsessive compulsive",
    "ptsd", "posttraumatic stress", "post-traumatic stress",
    "suicidal", "suicide", "self-harm",
]

BRAIN_LABELS = [
    "mri", "ct", "head", "brain", "neuro", "radiology",
]


def main():
    print("Loading MedicalDatasetCreator ...")
    mdc = MedicalDatasetCreator(directory="medical_datasets")

    filters = [
        # Adults only — comment out this line to include all ages
        {"field": "min_age", "string_list": ["18"]},
        # Case narrative mentions a psychiatric term
        {"field": "case_strings", "string_list": PSYCH_TERMS, "operator": "any"},
        # OR image caption mentions a psychiatric term
        {"field": "caption", "string_list": PSYCH_TERMS, "operator": "any"},
        # AND image label suggests brain / head imaging
        {"field": "label", "string_list": BRAIN_LABELS, "operator": "any"},
    ]

    print("Creating psych_brain_multimodal subset ...")
    print(f"  Psychiatric terms : {len(PSYCH_TERMS)} terms")
    print(f"  Brain image labels: {BRAIN_LABELS}")

    mdc.create_dataset(
        dataset_name="psych_brain_multimodal",
        filter_list=filters,
        dataset_type="multimodal",
    )

    print("Subset created at ./medical_datasets/psych_brain_multimodal/")
    print("Next step: inspect the directory, then run scripts/03_inspect_schema.py")


if __name__ == "__main__":
    main()
