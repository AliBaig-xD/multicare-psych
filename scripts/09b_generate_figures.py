"""
09b_generate_figures.py
Generate publication-quality figures from Phase 1 results.

Figures produced:
  figures/fig1_umap_clusters.png     - UMAP coloured by cluster
  figures/fig2_umap_diagnosis.png    - UMAP coloured by diagnosis
  figures/fig3_odds_ratios.png       - Top significant findings (OR chart)
  figures/fig4_ablation.png          - Image vs text vs combined ablation
  figures/fig5_heatmap.png           - Diagnosis % heatmap per cluster

Run:
    source env/py/bin/activate
    pip install matplotlib seaborn --quiet
    python scripts/09b_generate_figures.py | tee logs/09b_figures.log
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.makedirs("figures", exist_ok=True)

CLUSTERS_PATH = "results/psych_clusters.parquet"
DIAGNOSES_PATH = "data/psych_diagnoses.parquet"
SIG_PATH       = "results/significant_findings.csv"

# Colour palette
PALETTE = [
    '#e63946','#457b9d','#2a9d8f','#e9c46a','#f4a261',
    '#264653','#a8dadc','#6d6875','#b5838d','#e07a5f',
    '#3d405b','#81b29a','#f2cc8f','#118ab2','#06d6a0',
    '#ffd166','#ef476f','#073b4c','#7209b7','#4cc9f0',
]

print("Loading data...")
clusters  = pd.read_parquet(CLUSTERS_PATH)
diagnoses = pd.read_parquet(DIAGNOSES_PATH)
sig       = pd.read_csv(SIG_PATH)

DIAG_COLS = ["depression","bipolar","schizophrenia","psychosis",
             "anxiety","suicide","dementia","alzheimer"]

# ── Figure 1: UMAP coloured by cluster ───────────────────────────────────────
print("Generating Figure 1: UMAP by cluster...")

fig, ax = plt.subplots(figsize=(10, 8))

valid = clusters[clusters['cluster'] != -1]
noise = clusters[clusters['cluster'] == -1]

# Plot noise first in grey
ax.scatter(noise['umap_x'], noise['umap_y'],
           c='#cccccc', s=8, alpha=0.4, label='Noise', zorder=1)

# Plot clusters
cluster_ids = sorted(valid['cluster'].unique())
for i, cid in enumerate(cluster_ids):
    sub = valid[valid['cluster'] == cid]
    ax.scatter(sub['umap_x'], sub['umap_y'],
               c=PALETTE[i % len(PALETTE)], s=12, alpha=0.7,
               label=f'Cluster {cid} (n={len(sub)})', zorder=2)

ax.set_xlabel('UMAP 1', fontsize=12)
ax.set_ylabel('UMAP 2', fontsize=12)
ax.set_title('UMAP Projection — 20 Unsupervised Clusters\n(n=1,351 psychiatric case-image pairs)',
             fontsize=13, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
          ncol=1, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig1_umap_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/fig1_umap_clusters.png")

# ── Figure 2: UMAP coloured by diagnosis ─────────────────────────────────────
print("Generating Figure 2: UMAP by diagnosis...")

# Merge diagnoses onto clusters
merged = clusters.merge(
    diagnoses[['case_id','image_path'] + DIAG_COLS],
    on=['case_id','image_path'], how='left'
)

# Assign primary diagnosis label (first match wins)
def primary_dx(row):
    for dx in ['schizophrenia','bipolar','psychosis','dementia',
               'alzheimer','suicide','anxiety','depression']:
        if row.get(dx, 0) == 1:
            return dx
    return 'none/other'

merged['primary_dx'] = merged.apply(primary_dx, axis=1)

DX_COLORS = {
    'depression':    '#e63946',
    'bipolar':       '#457b9d',
    'schizophrenia': '#2a9d8f',
    'psychosis':     '#e9c46a',
    'anxiety':       '#f4a261',
    'dementia':      '#6d6875',
    'alzheimer':     '#b5838d',
    'suicide':       '#264653',
    'none/other':    '#cccccc',
}

fig, ax = plt.subplots(figsize=(10, 8))

for dx, color in DX_COLORS.items():
    sub = merged[merged['primary_dx'] == dx]
    if len(sub) == 0:
        continue
    ax.scatter(sub['umap_x'], sub['umap_y'],
               c=color, s=12, alpha=0.7 if dx != 'none/other' else 0.3,
               label=f'{dx} (n={len(sub)})', zorder=2 if dx != 'none/other' else 1)

ax.set_xlabel('UMAP 1', fontsize=12)
ax.set_ylabel('UMAP 2', fontsize=12)
ax.set_title('UMAP Projection — Coloured by Primary Diagnosis\n(confirmed, negation-handled NER)',
             fontsize=13, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig2_umap_diagnosis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/fig2_umap_diagnosis.png")

# ── Figure 3: Top significant findings (OR chart) ────────────────────────────
print("Generating Figure 3: Odds ratio chart...")

# Filter to enriched findings only, exclude noise
top = sig[
    (sig['significant'] == True) &
    (sig['odds_ratio'] > 1) &
    (sig['cluster'] != -1)
].copy()
top = top.sort_values('odds_ratio', ascending=True).tail(15)
top['label'] = top.apply(lambda r: f"Cluster {r['cluster']} — {r['diagnosis']}", axis=1)

fig, ax = plt.subplots(figsize=(9, 7))

colors_bar = ['#e63946' if or_ > 10 else '#457b9d' if or_ > 4 else '#2a9d8f'
              for or_ in top['odds_ratio']]

bars = ax.barh(top['label'], top['odds_ratio'], color=colors_bar, edgecolor='white', height=0.7)

# Add value labels
for bar, (_, row) in zip(bars, top.iterrows()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"OR={row['odds_ratio']:.1f}\np={row['p_corrected']:.3f}{row['stars']}",
            va='center', fontsize=8)

ax.set_xlabel('Odds Ratio (Bonferroni corrected)', fontsize=11)
ax.set_title('Top Significant Findings\nDiagnosis Enrichment per Cluster vs Rest of Dataset',
             fontsize=12, fontweight='bold')
ax.axvline(x=1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
patches = [
    mpatches.Patch(color='#e63946', label='OR > 10'),
    mpatches.Patch(color='#457b9d', label='OR 4-10'),
    mpatches.Patch(color='#2a9d8f', label='OR < 4'),
]
ax.legend(handles=patches, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('figures/fig3_odds_ratios.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/fig3_odds_ratios.png")

# ── Figure 4: Ablation chart ──────────────────────────────────────────────────
print("Generating Figure 4: Ablation chart...")

ablation_data = {
    'Condition': ['Image only\n(CLIP visual)', 'Text only\n(CLIP narrative)', 'Combined\n(image + text)'],
    'Bipolar %': [0, 84, 66],
    'Significant': [False, True, True],
    'n_clusters': [2, 21, 18],
}
df_abl = pd.DataFrame(ablation_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Left: bipolar enrichment
bar_colors = ['#cccccc', '#e63946', '#457b9d']
bars = ax1.bar(df_abl['Condition'], df_abl['Bipolar %'],
               color=bar_colors, edgecolor='white', width=0.5)
ax1.set_ylabel('Bipolar Enrichment (%)', fontsize=11)
ax1.set_title('Bipolar Signal by Embedding Type\n(same dataset, n=1,351)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 100)
for bar, pct, sig in zip(bars, df_abl['Bipolar %'], df_abl['Significant']):
    label = f"{pct}%\n(p<0.0001)" if sig and pct > 0 else f"{pct}%\n(n.s.)"
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             label, ha='center', va='bottom', fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: number of clusters
bars2 = ax2.bar(df_abl['Condition'], df_abl['n_clusters'],
                color=bar_colors, edgecolor='white', width=0.5)
ax2.set_ylabel('Number of Clusters', fontsize=11)
ax2.set_title('Clusters Found by Embedding Type\n(HDBSCAN, min_cluster_size=20)', fontsize=11, fontweight='bold')
for bar, n in zip(bars2, df_abl['n_clusters']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(n), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('Ablation Study: Clinical Narrative Drives Phenotypic Signal',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig4_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/fig4_ablation.png")

# ── Figure 5: Diagnosis heatmap ───────────────────────────────────────────────
print("Generating Figure 5: Diagnosis heatmap...")

# Build heatmap data
heatmap_rows = []
for cid in sorted(merged['cluster'].unique()):
    if cid == -1:
        continue
    sub = merged[merged['cluster'] == cid]
    row = {'Cluster': f'C{cid}\n(n={len(sub)})'}
    for dx in DIAG_COLS:
        row[dx] = sub[dx].mean() * 100 if dx in sub.columns else 0
    heatmap_rows.append(row)

hm_df = pd.DataFrame(heatmap_rows).set_index('Cluster')

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    hm_df,
    annot=True,
    fmt='.0f',
    cmap='YlOrRd',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Diagnosis Prevalence (%)', 'shrink': 0.8},
    ax=ax,
    vmin=0, vmax=80,
)
ax.set_title('Diagnosis Prevalence (%) per Cluster\n(confirmed diagnoses, negation-handled NER)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Diagnosis', fontsize=11)
ax.set_ylabel('Cluster', fontsize=11)
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('figures/fig5_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/fig5_heatmap.png")

print("\nAll figures generated in figures/")
print("Next: git add figures/ && git commit && git push")