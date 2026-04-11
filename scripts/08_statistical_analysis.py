"""
08_statistical_analysis.py
Statistical validation of psychiatric diagnosis enrichment per cluster.

For each cluster × diagnosis combination:
  - Chi-square test vs rest of dataset
  - Effect size (odds ratio)
  - Bonferroni-corrected p-values

Output:
  results/statistical_summary.csv
  results/significant_findings.csv   (p_corrected < 0.05 only)

Run:
    source env/py/bin/activate
    python scripts/08_statistical_analysis.py | tee logs/08_stats.log
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

DIAG_PATH = "data/psych_diagnoses.parquet"
OUT_CSV   = "results/statistical_summary.csv"
SIG_CSV   = "results/significant_findings.csv"

DIAG_COLS = [
    "depression", "bipolar", "schizophrenia", "psychosis",
    "anxiety", "ptsd", "ocd", "suicide", "dementia", "alzheimer"
]


def odds_ratio(a, b, c, d):
    """Compute odds ratio with continuity correction."""
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def main():
    df = pd.read_parquet(DIAG_PATH)
    clusters = sorted(c for c in df["cluster"].unique() if c != -1)

    rows = []
    for cluster_id in clusters:
        sub  = df[df["cluster"] == cluster_id]
        rest = df[df["cluster"] != cluster_id]
        imaging = sub["image_subtype"].value_counts().head(2).to_dict()

        for col in DIAG_COLS:
            a = int(sub[col].sum())
            b = int(len(sub) - a)
            c = int(rest[col].sum())
            d = int(len(rest) - c)

            if a == 0:
                continue

            _, p, _, _ = chi2_contingency([[a, b], [c, d]])
            or_val = odds_ratio(a, b, c, d)
            pct_cluster = a / len(sub) * 100
            pct_rest    = c / len(rest) * 100

            rows.append({
                "cluster":      cluster_id,
                "n_cluster":    len(sub),
                "imaging":      str(imaging),
                "diagnosis":    col,
                "n_positive":   a,
                "pct_cluster":  round(pct_cluster, 1),
                "pct_rest":     round(pct_rest, 1),
                "odds_ratio":   round(or_val, 2),
                "p_value":      p,
            })

    results = pd.DataFrame(rows)

    # Bonferroni correction
    n_tests = len(results)
    results["p_corrected"] = (results["p_value"] * n_tests).clip(upper=1.0)
    results["significant"] = results["p_corrected"] < 0.05
    results["stars"] = results["p_corrected"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    )

    results = results.sort_values(["cluster", "p_corrected"])
    results.to_csv(OUT_CSV, index=False)

    sig = results[results["significant"]].copy()
    sig.to_csv(SIG_CSV, index=False)

    print(f"Total tests       : {n_tests}")
    print(f"Significant (corr): {len(sig)}")
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {SIG_CSV}")

    print("\n=== SIGNIFICANT FINDINGS (Bonferroni corrected) ===")
    for _, row in sig.iterrows():
        direction = "ENRICHED" if row["pct_cluster"] > row["pct_rest"] else "DEPLETED"
        print(f"Cluster {row['cluster']:2d} | {row['diagnosis']:15s} | "
              f"{row['pct_cluster']:5.1f}% vs {row['pct_rest']:5.1f}% rest | "
              f"OR={row['odds_ratio']:5.2f} | p_corr={row['p_corrected']:.4f} {row['stars']} | "
              f"{direction}")


if __name__ == "__main__":
    main()
