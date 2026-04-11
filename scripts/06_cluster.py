"""
06_cluster.py
Phase D: Dimensionality reduction (UMAP) + density clustering (HDBSCAN).

Pipeline:
  1. Load 1024-dim embeddings
  2. StandardScaler normalisation
  3. UMAP → 2D
  4. HDBSCAN cluster assignment
  5. Merge cluster labels back onto the master table

Output:
  results/psych_clusters.parquet

Run:
    source env/py/bin/activate
    python scripts/06_cluster.py | tee logs/06_clusters.log
"""

import os

import hdbscan
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

EMB_PATH    = "data/psych_brain_embeddings.parquet"
MASTER_PATH = "data/psych_brain_master.parquet"
OUT_PATH    = "results/psych_clusters.parquet"

# ── Tunable hyperparameters ───────────────────────────────────────────────────
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.05
UMAP_METRIC       = "cosine"
UMAP_RANDOM_STATE = 42

HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES      = 5
# ─────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading embeddings ...")
    emb_df    = pd.read_parquet(EMB_PATH)
    master_df = pd.read_parquet(MASTER_PATH)

    orig_indices = emb_df["orig_index"].values
    X            = emb_df.drop(columns=["orig_index"]).values
    print(f"Embeddings shape: {X.shape}")

    print("Scaling ...")
    X_scaled = StandardScaler().fit_transform(X)

    print("Running UMAP ...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    X_umap = reducer.fit_transform(X_scaled)
    print(f"UMAP output shape: {X_umap.shape}")

    print("Running HDBSCAN ...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X_umap)

    # -1 = noise / unclustered points (HDBSCAN convention)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).sum() / len(labels) * 100
    print(f"\nClusters found (excl. noise): {n_clusters}")
    print(f"Noise points               : {(labels == -1).sum()} ({noise_pct:.1f}%)")

    # Merge back onto master
    result = master_df.iloc[orig_indices].copy()
    result["cluster"] = labels
    result["umap_x"]  = X_umap[:, 0]
    result["umap_y"]  = X_umap[:, 1]
    result = result.reset_index(drop=True)

    result.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")

    print("\nCluster size breakdown:")
    print(result["cluster"].value_counts().to_string())
    print("\nNext step: run the Streamlit atlas")
    print("  streamlit run app/atlas.py --server.port 8501 --server.address 0.0.0.0")


if __name__ == "__main__":
    main()
