"""
01_download_multicare.py
Phase A – Step 1: Download and index MultiCaRe v2.0 from Zenodo.

This script just initialises MedicalDatasetCreator, which triggers the
download and local indexing of the full MultiCaRe dataset into
./medical_datasets/.  It can take several minutes depending on bandwidth.

Run:
    source env/py/bin/activate
    python scripts/01_download_multicare.py | tee logs/01_download.log
"""

from multiversity.multicare_dataset import MedicalDatasetCreator


def main():
    print("Initialising MedicalDatasetCreator — this will download MultiCaRe v2.0 ...")
    mdc = MedicalDatasetCreator(directory="medical_datasets")
    print("Done. MultiCaRe is available at ./medical_datasets/")
    print("Next step: run scripts/02_build_psych_subset.py")


if __name__ == "__main__":
    main()
