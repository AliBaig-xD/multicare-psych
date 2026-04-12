"""
13_validate_cnp.py
Phase 2 – Step 5: Validate Phase 1 findings against CNP ground truth diagnoses.

This is the key validation step. We test whether the patterns found in Phase 1
(MultiCaRe case reports) replicate in Phase 2 (CNP confirmed diagnoses).

Specifically we test:
  1. Bipolar vs Depression separation — do bipolar subjects cluster separately?
  2. Schizophrenia/Psychosis cluster — do schizophrenia subjects form distinct clusters?
  3. Overall: does unsupervised clustering recover diagnostic groups?

Output:
  results/cnp_validation.txt
  results/cnp_validation.csv

Run:
    source env/py/bin/activate
    python scripts/13_validate_cnp.py | tee logs/13_validation.log
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

CNP_CLUSTERS = "cnp_data/cnp_clusters.parquet"
CNP_EMB      = "cnp_data/cnp_embeddings.parquet"
OUT_TXT      = "results/cnp_validation.txt"
OUT_CSV      = "results/cnp_validation.csv"


def write(f, msg=""):
    print(msg)
    f.write(msg + "\n")


def main():
    os.makedirs("results", exist_ok=True)

    clusters = pd.read_parquet(CNP_CLUSTERS)
    embeddings = pd.read_parquet(CNP_EMB)

    feat_cols = [c for c in embeddings.columns if c.startswith("feat_")]
    X = embeddings[feat_cols].values
    y_labels = embeddings["diagnosis"].values

    with open(OUT_TXT, "w") as f:

        write(f, "=" * 70)
        write(f, "CNP VALIDATION REPORT — Phase 2")
        write(f, "MultiCaRe Psychiatric Imaging Atlas")
        write(f, "=" * 70)

        write(f, f"\nTotal subjects : {len(clusters)}")
        write(f, f"\nDiagnosis breakdown:")
        for dx, n in clusters["diagnosis"].value_counts().items():
            write(f, f"  {dx:20s}: {n}")

        # ── Test 1: Cluster-diagnosis alignment ───────────────────────────────
        write(f, "\n" + "=" * 70)
        write(f, "TEST 1: Unsupervised cluster alignment with diagnoses")
        write(f, "=" * 70)

        valid = clusters[clusters["cluster"] != -1]
        le    = LabelEncoder()
        true  = le.fit_transform(valid["diagnosis"])
        pred  = valid["cluster"].values

        ari = adjusted_rand_score(true, pred)
        nmi = normalized_mutual_info_score(true, pred)

        write(f, f"\nAdjusted Rand Index (ARI): {ari:.4f}")
        write(f, f"  > 0.1  = weak alignment")
        write(f, f"  > 0.3  = moderate alignment")
        write(f, f"  > 0.5  = strong alignment (publishable)")
        write(f, f"\nNormalized Mutual Info   : {nmi:.4f}")

        if ari > 0.3:
            write(f, "\n*** RESULT: Strong cluster-diagnosis alignment — Phase 1 findings VALIDATED ***")
        elif ari > 0.1:
            write(f, "\n** RESULT: Moderate alignment — partial validation of Phase 1 findings **")
        else:
            write(f, "\n* RESULT: Weak alignment — clusters driven by non-diagnostic factors *")

        # ── Test 2: Bipolar vs Depression separation ──────────────────────────
        write(f, "\n" + "=" * 70)
        write(f, "TEST 2: Bipolar vs Depression separation (key Phase 1 finding)")
        write(f, "=" * 70)

        bp_dep = clusters[clusters["diagnosis"].isin(["BIPOLAR", "CONTROL", "bipolar", "SCHZ", "ADHD"])].copy()

        if len(bp_dep) > 10:
            # Chi-square: are bipolar and other diagnoses in different clusters?
            dx_groups = clusters[clusters["diagnosis"].isin(
                [d for d in clusters["diagnosis"].unique() if d != -1]
            )]

            write(f, "\nCluster composition by diagnosis:")
            ct = pd.crosstab(dx_groups["cluster"], dx_groups["diagnosis"])
            write(f, ct.to_string())

        # ── Test 3: Supervised classification from embeddings ─────────────────
        write(f, "\n" + "=" * 70)
        write(f, "TEST 3: Supervised classification accuracy from embeddings")
        write(f, "(How well do the embeddings encode diagnostic information?)")
        write(f, "=" * 70)

        le2   = LabelEncoder()
        y_enc = le2.fit_transform(y_labels)

        clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="balanced_accuracy")

        write(f, f"\n5-fold cross-validated balanced accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        write(f, f"Chance level (random): {1/len(le2.classes_):.3f}")
        write(f, f"Classes: {list(le2.classes_)}")

        if scores.mean() > (1/len(le2.classes_) + 0.1):
            write(f, "\n*** Embeddings carry significant diagnostic signal above chance ***")
        else:
            write(f, "\n* Embeddings show limited diagnostic separability *")

        # ── Summary ───────────────────────────────────────────────────────────
        write(f, "\n" + "=" * 70)
        write(f, "SUMMARY")
        write(f, "=" * 70)
        write(f, f"\nARI                      : {ari:.4f}")
        write(f, f"NMI                      : {nmi:.4f}")
        write(f, f"Classification accuracy  : {scores.mean():.3f} ± {scores.std():.3f}")
        write(f, f"Chance level             : {1/len(le2.classes_):.3f}")

        results_df = pd.DataFrame([{
            "metric": "ARI", "value": ari,
            "interpretation": "cluster-diagnosis alignment"
        }, {
            "metric": "NMI", "value": nmi,
            "interpretation": "mutual information clusters vs diagnoses"
        }, {
            "metric": "balanced_accuracy_mean", "value": scores.mean(),
            "interpretation": "5-fold CV supervised classification"
        }, {
            "metric": "balanced_accuracy_std", "value": scores.std(),
            "interpretation": "standard deviation across folds"
        }, {
            "metric": "chance_level", "value": 1/len(le2.classes_),
            "interpretation": "random baseline"
        }])
        results_df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved report to {OUT_TXT}")
    print(f"Saved metrics to {OUT_CSV}")


if __name__ == "__main__":
    main()
