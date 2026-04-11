# Phase 1 Findings: MultiCaRe Psychiatric Imaging Atlas

**Author:** Ali Baig (Independent Researcher)  
**Date:** April 2026  
**Status:** Preliminary — Phase 1 complete, external validation pending

---

## Overview

We applied unsupervised multimodal clustering to 1,351 brain-related psychiatric
case reports from the MultiCaRe dataset. Each case was represented as a combined
embedding of its brain image and clinical text using CLIP (Contrastive
Language-Image Pretraining). UMAP dimensionality reduction followed by HDBSCAN
density clustering produced 20 clusters. Psychiatric diagnoses were then extracted
using scispacy NER with negspacy negation handling — distinguishing confirmed
diagnoses from negated mentions ("no history of depression") and family history.
Cluster enrichment was validated using chi-square tests with Bonferroni correction.

---

## Dataset

- Source: MultiCaRe v2.0 (PubMed Central open-access case reports)
- Initial filter: adult psychiatric cases with brain/head imaging labels
- After path resolution and quality filtering: **1,351 case-image pairs**
- Imaging modalities: MRI (n=846), CT (n=378), PET (n=57), SPECT (n=38), X-ray (n=32)
- All radiology region: head

---

## Methods Summary

| Step | Tool | Parameters |
|------|------|------------|
| Image + text embedding | CLIP ViT-B/32 | 512-dim each, concatenated → 1024-dim |
| Dimensionality reduction | UMAP | n_neighbors=15, min_dist=0.05, cosine metric |
| Clustering | HDBSCAN | min_cluster_size=20, min_samples=5 |
| Diagnosis extraction | scispacy en_core_sci_sm + negspacy | First 3000 chars per case |
| Statistical test | Chi-square vs rest | Bonferroni corrected, α=0.05 |

---

## Significant Findings (Bonferroni Corrected)

### Finding 1: Bipolar MRI Phenotype (Clusters 7 & 8)

The two largest bipolar-enriched clusters both consist predominantly of structural
MRI scans and show simultaneous depletion of depression diagnoses.

| Cluster | n | Diagnosis | % in cluster | % in rest | Odds Ratio | p (corrected) |
|---------|---|-----------|-------------|-----------|------------|---------------|
| 8 | 109 | bipolar | 66.1% | 23.9% | 6.14 | <0.001 *** |
| 8 | 109 | depression | 13.8% | 42.3% | 0.22 | <0.001 *** |
| 7 | 39 | bipolar | 59.0% | 26.4% | 3.97 | 0.002 ** |
| 7 | 39 | depression | 10.3% | 40.9% | 0.18 | 0.026 * |

**Interpretation:** The algorithm separated bipolar disorder cases from depression
cases on the basis of brain MRI appearance and clinical text, without any
supervision. This replicates a known but debated clinical observation that bipolar
disorder has distinct structural brain correlates compared to major depressive
disorder, including differences in white matter integrity and subcortical volumes.

---

### Finding 2: Depression-Dementia Functional Imaging Cluster (Cluster 9)

| Cluster | n | Diagnosis | % in cluster | % in rest | Odds Ratio | p (corrected) |
|---------|---|-----------|-------------|-----------|------------|---------------|
| 9 | 61 | depression | 70.5% | 38.5% | 3.75 | <0.001 *** |
| 9 | 61 | dementia | 24.6% | 2.2% | 14.25 | <0.001 *** |
| 9 | 61 | ptsd | 4.9% | 0.2% | 22.01 | 0.001 ** |
| 9 | 61 | alzheimer | 9.8% | 1.8% | 6.32 | 0.017 * |
| 9 | 61 | bipolar | 4.9% | 28.4% | 0.15 | 0.012 * |

Imaging breakdown: PET (n=21), SPECT (n=17), MRI (n=17)

**Interpretation:** This cluster captures cases at the depression-dementia
diagnostic interface. PET and SPECT functional imaging is clinically used
precisely to distinguish treatment-resistant depression from early-onset dementia,
as both can present with similar symptoms. The co-occurrence of depression and
dementia diagnoses with functional imaging modality in a single cluster — without
supervision — is clinically coherent and suggests these cases share a distinct
multimodal signature. The PTSD enrichment (OR=22.01) is unexpected and warrants
further investigation.

---

### Finding 3: Severe Psychiatric MRI Phenotype (Cluster 17)

| Cluster | n | Diagnosis | % in cluster | % in rest | Odds Ratio | p (corrected) |
|---------|---|-----------|-------------|-----------|------------|---------------|
| 17 | 89 | psychosis | 34.8% | 6.8% | 7.32 | <0.001 *** |
| 17 | 89 | suicide | 23.6% | 6.3% | 4.67 | <0.001 *** |
| 17 | 89 | schizophrenia | 12.4% | 3.4% | 4.11 | 0.011 * |
| 17 | 89 | anxiety | 34.8% | 18.6% | 2.35 | 0.039 * |

Imaging: MRI dominant (n=78/89)

**Interpretation:** This cluster represents the severe end of the psychiatric
spectrum — psychosis, schizophrenia, and high suicide risk co-occurring in
structural MRI cases. The combination of psychosis + suicide risk enrichment is
clinically significant as psychotic disorders carry substantially elevated suicide
risk compared to other psychiatric conditions.

---

### Finding 4: Alzheimer-Anxiety Cluster (Cluster 0)

| Cluster | n | Diagnosis | % in cluster | % in rest | Odds Ratio | p (corrected) |
|---------|---|-----------|-------------|-----------|------------|---------------|
| 0 | 20 | alzheimer | 45.0% | 1.5% | 52.85 | <0.001 *** |
| 0 | 20 | anxiety | 40.0% | 18.5% | 2.89 | 0.044 * |

**Interpretation:** Small but extremely pure cluster. Alzheimer's presenting with
anxiety is a recognised clinical pattern in early-stage dementia — patients often
present with anxiety before cognitive deficits become apparent. The OR of 52.85
is the highest in the entire dataset.

---

### Finding 5: Schizophrenia/Psychosis MRI Clusters (Clusters 2 & 19)

| Cluster | n | Diagnosis | % in cluster | % in rest | Odds Ratio | p (corrected) |
|---------|---|-----------|-------------|-----------|------------|---------------|
| 2 | 24 | schizophrenia | 25.0% | 3.6% | 9.27 | <0.001 *** |
| 19 | 76 | psychosis | 30.3% | 7.4% | 5.49 | <0.001 *** |

**Interpretation:** Two separate MRI clusters enriched for psychotic disorders,
distinct from Cluster 17. May represent different phases or presentations of
psychotic illness.

---

## Limitations

1. **Dataset bias:** MultiCaRe is derived from published case reports. Journals
   preferentially publish unusual or instructive cases, not routine presentations.
   Findings may not generalise to the broader clinical population.

2. **Sample size:** 1,351 cases total, with individual clusters as small as 20.
   Statistical power is limited for smaller clusters.

3. **Modality confound:** Imaging modality (MRI vs CT vs PET) partially drives
   clustering. The bipolar finding (Clusters 7/8) and psychosis findings (Clusters
   17/19) are within the same modality (MRI) so are less confounded. The
   depression-dementia cluster (Cluster 9) is partially modality-driven (PET/SPECT).

4. **NER limitations:** scispacy entity extraction on clinical narratives is
   imperfect. Negation handling covers common patterns but complex clinical language
   may be misclassified.

5. **No external validation:** All findings are from a single dataset. Replication
   on an independent dataset is required before any clinical interpretation.

---

## Next Steps (Phase 2)

Validate findings on the Consortium for Neuropsychiatric Phenomics (CNP) dataset
(OpenNeuro ds000030) which contains:
- Confirmed DSM diagnoses (bipolar disorder, schizophrenia, ADHD, healthy controls)
- Structural and functional MRI scans
- ~272 subjects across diagnostic groups

If the bipolar vs depression separation (Finding 1) and psychosis cluster
(Finding 3) replicate on CNP with confirmed diagnoses and raw MRI data, that
constitutes meaningful external validation.

---

## Raw Results

See [`results/significant_findings.csv`](results/significant_findings.csv) for
all Bonferroni-corrected significant findings and
[`results/statistical_summary.csv`](results/statistical_summary.csv) for the
complete statistical table.