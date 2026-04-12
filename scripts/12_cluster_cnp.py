"""
12_cluster_cnp.py
Phase 2 – Step 4: Cluster CNP embeddings and evaluate against confirmed diagnoses.

Unlike Phase 1 (unsupervised only), here we have ground truth labels so we can:
  - Run unsupervised clustering
  - Measure how well clusters align with confirmed diagnoses
  - Compute ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information)
  - Visualise UMAP coloured by diagnosis vs cluster

Output:
  cnp_data/cnp_clusters.parquet
  results/cnp_cluster_summary.csv

Run:
    source env/py/bin/activate
    python scripts/12_cluster_cnp.py | tee logs/12_cluster_cnp.log
"""

import os
import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score

EMB_PATH  = "cnp_data/cnp_embeddings.parquet"
OUT_PATH  = "cnp_data/cnp_clusters.parquet"
SUM_PATH  = "results/cnp_cluster_summary.csv"

UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.05
UMAP_METRIC       = "cosine"
UMAP_RANDOM_STATE = 42

HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES      = 3


def main():
    os.makedirs("results", exist_ok=True)

    df = pd.read_parquet(EMB_PATH)
    print(f"Subjects: {len(df)}")
    print(f"Diagnosis breakdown:\n{df['diagnosis'].value_counts().to_string()}")

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feat_cols].values
    print(f"\nEmbedding dim: {X.shape[1]}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    X_scaled = StandardScaler().fit_transform(X)

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    X_umap = reducer.fit_transform(X_scaled)

    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(X_umap)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_pct  = (cluster_labels == -1).sum() / len(cluster_labels) * 100
    print(f"\nClusters found : {n_clusters}")
    print(f"Noise          : {(cluster_labels == -1).sum()} ({noise_pct:.1f}%)")

    # ── Evaluate against ground truth ─────────────────────────────────────────
    valid_mask = cluster_labels != -1
    if valid_mask.sum() > 0:
        true_labels = LabelEncoder().fit_transform(df["diagnosis"])
        ari = adjusted_rand_score(true_labels[valid_mask], cluster_labels[valid_mask])
        nmi = normalized_mutual_info_score(true_labels[valid_mask], cluster_labels[valid_mask])
        print(f"\nAdjusted Rand Index (ARI): {ari:.4f}  (1.0 = perfect, 0 = random)")
        print(f"Normalized Mutual Info    : {nmi:.4f}  (1.0 = perfect, 0 = random)")

        if valid_mask.sum() > n_clusters:
            sil = silhouette_score(X_umap[valid_mask], cluster_labels[valid_mask])
            print(f"Silhouette Score          : {sil:.4f}  (1.0 = perfect separation)")

    # ── Per-cluster diagnosis breakdown ───────────────────────────────────────
    result = df.copy()
    result["cluster"] = cluster_labels
    result["umap_x"]  = X_umap[:, 0]
    result["umap_y"]  = X_umap[:, 1]

    print("\nCluster × Diagnosis breakdown:")
    summary_rows = []
    for c in sorted(result["cluster"].unique()):
        sub = result[result["cluster"] == c]
        counts = sub["diagnosis"].value_counts().to_dict()
        dominant = max(counts, key=counts.get)
        dominant_pct = counts[dominant] / len(sub) * 100
        print(f"  Cluster {c:2d} ({len(sub):3d} subjects) | dominant: {dominant} {dominant_pct:.0f}%  | {counts}")
        summary_rows.append({
            "cluster":      c,
            "n":            len(sub),
            "dominant_dx":  dominant,
            "dominant_pct": round(dominant_pct, 1),
            **{f"n_{k}": v for k, v in counts.items()}
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUM_PATH, index=False)

    # Save full results
    result_out = result[["sub_id", "diagnosis", "age", "gender", "cluster", "umap_x", "umap_y"]].copy()
    result_out.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved clusters to {OUT_PATH}")
    print(f"Saved summary  to {SUM_PATH}")
    print("\nNext step: run scripts/13_validate_cnp.py")


if __name__ == "__main__":
    main()
