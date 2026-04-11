"""
app/atlas.py
Phase E: Interactive Streamlit atlas for exploring psychiatric imaging clusters.

Access via SSH tunnel from your local machine:
  ssh -L 8501:localhost:8501 ubuntu@<your_vps_ip>

Then open: http://localhost:8501

Run on VPS:
    source env/py/bin/activate
    streamlit run app/atlas.py --server.port 8501 --server.address 0.0.0.0
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

REPO_ROOT      = "/home/ubuntu/multicare-psych"
CLUSTER_PATH   = "results/psych_clusters.parquet"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Psychiatric Imaging Atlas",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(CLUSTER_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 MultiCaRe Psych Atlas")
st.sidebar.markdown(
    "Unsupervised clusters of brain-related psychiatric case reports "
    "from the MultiCaRe dataset."
)

df = load_data()

all_clusters = sorted(df["cluster"].unique())
# Put noise cluster (-1) at the end if present
ordered = [c for c in all_clusters if c != -1] + ([-1] if -1 in all_clusters else [])

selected_cluster = st.sidebar.selectbox(
    "Select cluster",
    ordered,
    format_func=lambda c: f"Cluster {c}  (noise)" if c == -1 else f"Cluster {c}",
)

n_show = st.sidebar.slider("Examples to display", min_value=1, max_value=50, value=10)

# ── Main panel — UMAP scatter ─────────────────────────────────────────────────
st.title("Psychiatric Imaging Cluster Atlas")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("UMAP projection — all clusters")
    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color=df["cluster"].astype(str),
        hover_data=["case_id", "image_labels"] if "case_id" in df.columns else ["image_labels"],
        labels={"color": "Cluster"},
        height=500,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Cluster sizes")
    counts = df["cluster"].value_counts().rename_axis("Cluster").reset_index(name="Cases")
    counts["Cluster"] = counts["Cluster"].astype(str)
    st.dataframe(counts, use_container_width=True)

# ── Selected cluster deep-dive ────────────────────────────────────────────────
st.divider()
cluster_df = df[df["cluster"] == selected_cluster]
st.subheader(
    f"Cluster {selected_cluster}  —  {len(cluster_df)} cases"
    + ("  (noise / unclustered)" if selected_cluster == -1 else "")
)

# UMAP zoom on selected cluster
fig2 = px.scatter(
    cluster_df,
    x="umap_x",
    y="umap_y",
    color_discrete_sequence=["#e63946"],
    height=300,
    title=f"Cluster {selected_cluster} — UMAP zoom",
)
fig2.update_traces(marker=dict(size=5, opacity=0.8))
st.plotly_chart(fig2, use_container_width=True)

# Case examples
st.subheader(f"Random sample of {n_show} cases")
sample = cluster_df.sample(min(n_show, len(cluster_df)), random_state=None)

for _, row in sample.iterrows():
    with st.expander(f"Case: {row.get('case_id', 'N/A')}  |  Labels: {row.get('image_labels', '')}"):
        img_col, text_col = st.columns([1, 2])

        with img_col:
            img_path = os.path.join(REPO_ROOT, str(row.get("image_path", "")))
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    st.image(img, caption=str(row.get("image_caption", "")), use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not load image: {e}")
            else:
                st.warning("Image file not found.")

        with text_col:
            st.markdown("**Caption**")
            st.write(str(row.get("image_caption", "—")))
            st.markdown("**Case text (first 800 chars)**")
            st.write(str(row.get("case_text", "—"))[:800])
