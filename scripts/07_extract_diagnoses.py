"""
07_extract_diagnoses.py
Extract confirmed psychiatric diagnoses from case texts using:
  - scispacy NER for entity detection
  - negspacy for negation handling
  - Rule-based filters for family history / rule-out language

This replaces naive keyword counting with proper clinical NLP:
  - "No history of depression"      → depression NOT counted
  - "Family history of bipolar"     → bipolar NOT counted
  - "Rule out schizophrenia"        → schizophrenia NOT counted
  - "Patient diagnosed with anxiety"→ anxiety COUNTED

Output:
  data/psych_diagnoses.parquet

Run:
    pip install negspacy  # one-time
    source env/py/bin/activate
    python scripts/07_extract_diagnoses.py | tee logs/07_diagnoses.log
"""

import pandas as pd
import spacy
import scispacy
from negspacy.negation import Negex
from tqdm import tqdm

MASTER_PATH = "data/psych_brain_master.parquet"
OUT_PATH    = "data/psych_diagnoses.parquet"

# Phrases that indicate NOT a current patient diagnosis
EXCLUDE_PREFIXES = [
    "family history", "rule out", "r/o",
    "no family history", "mother had", "father had",
    "history of family", "sibling", "parent",
]


def is_family_or_ruleout(text: str, ent_start: int) -> bool:
    """Check if entity is preceded by family history or rule-out language."""
    preceding = text[:ent_start].lower().strip()[-40:]
    return any(phrase in preceding for phrase in EXCLUDE_PREFIXES)


def extract_diagnoses(text: str, nlp) -> dict:
    """Extract confirmed psychiatric diagnoses from a case text."""
    result = {term: 0 for term in [
        "depression", "bipolar", "schizophrenia", "psychosis",
        "anxiety", "ptsd", "ocd", "suicide", "dementia", "alzheimer"
    ]}

    # Process first 3000 chars per case for speed
    doc = nlp(text[:3000])

    for ent in doc.ents:
        ent_lower = ent.text.lower()

        # Skip negated entities
        if ent._.negex:
            continue

        # Skip family history / rule-out
        if is_family_or_ruleout(text, ent.start_char):
            continue

        # Map to diagnosis categories
        if any(t in ent_lower for t in ["depress", "mdd"]):
            result["depression"] = 1
        if any(t in ent_lower for t in ["bipolar", "mania", "manic"]):
            result["bipolar"] = 1
        if "schizophreni" in ent_lower:
            result["schizophrenia"] = 1
        if any(t in ent_lower for t in ["psychos", "psychotic"]):
            result["psychosis"] = 1
        if any(t in ent_lower for t in ["anxiety", "panic"]):
            result["anxiety"] = 1
        if any(t in ent_lower for t in ["ptsd", "posttraumatic", "post-traumatic"]):
            result["ptsd"] = 1
        if any(t in ent_lower for t in ["ocd", "obsessive"]):
            result["ocd"] = 1
        if "suicid" in ent_lower:
            result["suicide"] = 1
        if "dementia" in ent_lower:
            result["dementia"] = 1
        if "alzheimer" in ent_lower:
            result["alzheimer"] = 1

    return result


def main():
    print("Loading scispacy model...")
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("negex", last=True)

    df = pd.read_parquet(MASTER_PATH)
    print(f"Processing {len(df)} cases...")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        diag = extract_diagnoses(str(row["case_text"]), nlp)
        diag["case_id"]    = row["case_id"]
        diag["image_path"] = row["image_path"]
        rows.append(diag)

    diag_df = pd.DataFrame(rows)

    # Merge with cluster assignments
    clusters = pd.read_parquet("results/psych_clusters.parquet")[[
        "case_id", "image_path", "cluster", "umap_x", "umap_y",
        "image_subtype", "radiology_view", "radiology_region",
        "case_text", "caption", "gender", "age",
    ]]
    merged = clusters.merge(diag_df, on=["case_id", "image_path"], how="left")
    merged.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(merged)} rows to {OUT_PATH}")
    print("\nDiagnosis prevalence (confirmed, non-negated):")
    diag_cols = ["depression", "bipolar", "schizophrenia", "psychosis",
                 "anxiety", "ptsd", "ocd", "suicide", "dementia", "alzheimer"]
    for col in diag_cols:
        n = merged[col].sum()
        pct = n / len(merged) * 100
        print(f"  {col:20s}: {n:4.0f} ({pct:.1f}%)")

    print("\nNext step: run scripts/08_statistical_analysis.py")


if __name__ == "__main__":
    main()
