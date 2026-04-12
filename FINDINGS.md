# Phase 1 & 2 Findings: MultiCaRe Psychiatric Imaging Atlas

**Author:** Ali Baig (Independent Researcher)  
**Date:** April 2026  
**Status:** Complete — preprint ready

---

## Summary

We applied unsupervised multimodal clustering to 1,351 psychiatric brain imaging
case reports from MultiCaRe, found statistically significant phenotypic clusters,
then determined through ablation and external validation that the signal is driven
by clinical narrative text rather than brain imaging features.

---

## Phase 1: MultiCaRe Clustering (n=1,351)

### Method
- CLIP ViT-B/32 image + text embeddings (1024-dim combined)
- UMAP (n_neighbors=15, min_dist=0.05) + HDBSCAN (min_cluster_size=20)
- scispacy + negspacy diagnosis extraction with negation handling
- Chi-square tests, Bonferroni corrected

### Significant Findings

| Cluster | n | Key diagnosis | OR | p (corrected) |
|---------|---|---------------|----|---------------|
| 8 | 109 | bipolar 66% | 6.14 | <0.001 *** |
| 8 | 109 | depression DEPLETED | 0.22 | <0.001 *** |
| 7 | 39 | bipolar 59% | 3.97 | 0.002 ** |
| 9 | 61 | depression 70% | 3.75 | <0.001 *** |
| 9 | 61 | dementia 25% | 14.25 | <0.001 *** |
| 9 | 61 | PTSD | 22.01 | 0.001 ** |
| 17 | 89 | psychosis 35% | 7.32 | <0.001 *** |
| 17 | 89 | suicide 24% | 4.67 | <0.001 *** |
| 0 | 20 | alzheimer 45% | 52.85 | <0.001 *** |
| 2 | 24 | schizophrenia 25% | 9.27 | <0.001 *** |
| 19 | 76 | psychosis 30% | 5.49 | <0.001 *** |

Full results: [`results/significant_findings.csv`](results/significant_findings.csv)

---

## Ablation: Image vs Text Signal

| Condition | Clusters | Bipolar enrichment |
|-----------|----------|--------------------|
| Image only | 2 | None |
| Text only | 21 | 84% bipolar, p<0.0001 |
| Combined | 18 | 59-66% bipolar, p<0.0001 |

**Conclusion: signal is text-driven, not imaging-driven.**

---

## Phase 2: CNP Validation (n=265, confirmed DSM diagnoses)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | -0.027 | Random |
| NMI | 0.062 | Minimal |
| Balanced accuracy | 0.250 | Exactly chance |

Raw MRI embeddings (CLIP + ROI) did not recover diagnostic groupings.
Confirms ablation finding: current image embeddings carry no psychiatric signal.

---

## Interpretation

1. Clinical case report text is a rich signal for psychiatric phenotyping
2. General-purpose image embeddings (CLIP) are insufficient for psychiatric neuroimaging
3. Specialized models (FreeSurfer, cortical thickness, brain foundation models) needed

---

## Limitations

- Case report dataset bias (unusual cases, not representative)
- Small sample (1,351 Phase 1, 265 Phase 2)
- CLIP not designed for medical imaging
- No prospective validation

---

## Files

- [`results/significant_findings.csv`](results/significant_findings.csv) — Bonferroni corrected findings
- [`results/statistical_summary.csv`](results/statistical_summary.csv) — full stats
- [`results/cnp_validation.csv`](results/cnp_validation.csv) — Phase 2 validation metrics
- [`PREPRINT.md`](PREPRINT.md) — full manuscript
