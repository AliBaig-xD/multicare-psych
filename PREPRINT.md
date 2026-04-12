# Unsupervised Clustering of Multimodal Psychiatric Case Reports Reveals Clinically Coherent Phenotypes Driven by Clinical Narrative

**Ali Baig**  
Independent Researcher  
Contact: [your email]

*Preprint — not peer reviewed*  
*April 2026*

---

## Abstract

Psychiatric diagnosis relies on clinical judgment rather than objective biomarkers, motivating computational approaches to phenotype discovery. We applied unsupervised multimodal clustering to 1,351 brain-related psychiatric case reports from the MultiCaRe dataset — a large open-access collection of PubMed Central clinical case reports containing both images and clinical text. Each case was embedded using CLIP (Contrastive Language-Image Pretraining) to produce combined image-text representations, reduced with UMAP, and clustered with HDBSCAN. Psychiatric diagnoses were extracted using clinical NER with negation handling (scispacy + negspacy) and cluster enrichment was validated with chi-square tests and Bonferroni correction. We identified 20 clusters, several with statistically significant diagnosis enrichment: a bipolar-enriched cluster (OR=6.14, p<0.001), a depression-dementia co-occurrence cluster with functional imaging (OR=14.25, p<0.001), and a severe psychosis-suicide cluster (OR=7.32, p<0.001).

To determine the source of this signal, we performed an ablation separating image and text embeddings. The bipolar separation was absent in image-only clustering but strongly present in text-only clustering (84% bipolar enrichment, p<0.0001), demonstrating that the observed phenotypes are driven by **clinical narrative** rather than imaging features. External validation on the Consortium for Neuropsychiatric Phenomics (CNP) dataset using raw T1w MRI embeddings confirmed this: ARI=-0.027, classification accuracy=0.250 (chance level).

These findings establish that multimodal clinical case reports are a rich resource for **narrative-driven psychiatric phenotyping**. They also highlight a critical gap: general-purpose image embeddings do not capture psychiatric neuroimaging signal, and specialized brain-specific models are needed before imaging can contribute meaningfully to unsupervised psychiatric subtyping.

---

## 1. Introduction

Psychiatric disorders affect over 1 billion people worldwide and represent a leading cause of disability. Despite decades of neuroimaging research, no reliable imaging biomarkers have been established for routine clinical use. Diagnosis relies on clinical interviews, symptom checklists, and clinician judgment, leading to high rates of misdiagnosis and delayed treatment.

Unsupervised machine learning applied to clinical data has shown promise in identifying psychiatric subtypes that cut across traditional diagnostic boundaries. Prior work has identified biotypes of depression using resting-state fMRI, subgroups of schizophrenia using structural MRI, and imaging-defined subtypes of Alzheimer's disease. However, these studies typically require large, carefully curated datasets with confirmed diagnoses, standardized imaging protocols, and institutional data access — barriers that limit broad participation in psychiatric research.

The MultiCaRe dataset offers an alternative starting point: a large open-access collection of over 93,000 clinical case reports from PubMed Central, each containing clinical narrative text, associated images, image captions, and metadata. While case reports are not representative of the general clinical population, they represent a rich and underutilised resource for hypothesis generation and methodology development.

In this work, we present an unsupervised multimodal clustering analysis of psychiatric case reports from MultiCaRe. We combine image and text embeddings, apply density-based clustering, extract confirmed psychiatric diagnoses using clinical NLP, and statistically validate cluster enrichment. Critically, we perform ablation experiments to determine the relative contribution of imaging versus narrative signal, and validate our approach on an independent dataset with confirmed DSM diagnoses.

Our central finding is that the observed phenotypic structure is driven by clinical narrative text, not brain imaging. We discuss the implications for both clinical NLP and psychiatric neuroimaging, and propose specific directions for future work using brain-specific models.

---

## 2. Methods

### 2.1 Dataset

We used MultiCaRe v2.0, downloaded via the official `multiversity` Python toolkit. The full dataset contains 93,000+ clinical cases and 130,000+ images derived from open-access PubMed Central articles.

We filtered to a psychiatric brain imaging subset using the following criteria:
- Adult cases (age ≥ 18)
- Case text OR image caption contains at least one psychiatric term (schizophrenia, bipolar, mania, depression, psychosis, anxiety, PTSD, OCD, suicidal, self-harm)
- Image label indicates brain or head imaging (MRI, CT, head, brain, neuro, radiology)

This yielded 5,449 psychiatric cases. After joining against full image metadata filtered to head/brain radiology and resolving file paths, our final dataset contained **1,351 case-image pairs** across **4,901 unique cases**.

Imaging modalities: MRI (n=846, 62.6%), CT (n=378, 28.0%), PET (n=57, 4.2%), SPECT (n=38, 2.8%), X-ray (n=32, 2.4%).

### 2.2 Multimodal Embedding

Each case-image pair was embedded using CLIP ViT-B/32. For each pair:

- **Image embedding:** Image processed through CLIP vision encoder → 512-dim L2-normalised vector
- **Text embedding:** First 1,000 chars of case text + first 500 chars of image caption → CLIP text encoder → 512-dim L2-normalised vector
- **Combined embedding:** Concatenated → 1,024-dim representation

### 2.3 Clustering

Combined embeddings were standardised (StandardScaler), reduced with UMAP (n_neighbors=15, min_dist=0.05, cosine metric, random_state=42), and clustered with HDBSCAN (min_cluster_size=20, min_samples=5), producing 20 clusters with 26.8% noise.

### 2.4 Diagnosis Extraction

Psychiatric diagnoses were extracted using scispacy (en_core_sci_sm) with negspacy negation detection. Ten diagnostic categories were extracted: depression, bipolar, schizophrenia, psychosis, anxiety, PTSD, OCD, suicide/suicidality, dementia, Alzheimer's disease. Negated mentions, family history, and rule-out language were excluded.

### 2.5 Statistical Validation

Chi-square tests compared diagnosis prevalence per cluster vs rest of dataset. Odds ratios with continuity correction were computed. Bonferroni correction applied (n_tests=200). Significance threshold: p_corrected < 0.05.

### 2.6 Ablation: Image vs Text Signal

To isolate the source of clustering signal, we re-ran UMAP + HDBSCAN separately on:
- Image-only embeddings (dims 0-511)
- Text-only embeddings (dims 512-1023)
- Combined embeddings (all 1,024 dims)

Bipolar cluster enrichment was compared across conditions.

### 2.7 External Validation (CNP Dataset)

We validated on the Consortium for Neuropsychiatric Phenomics (CNP, OpenNeuro ds000030): 265 subjects with confirmed DSM diagnoses (125 controls, 50 schizophrenia, 49 bipolar, 41 ADHD).

T1w structural MRI scans were downloaded, axial/sagittal/coronal slices extracted, and embedded with CLIP. A 48-dimensional ROI brain fingerprint was extracted using the Harvard-Oxford cortical atlas (nilearn). Combined 560-dim embeddings were clustered with the same pipeline.

Evaluation: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and 5-fold cross-validated logistic regression balanced accuracy.

---

## 3. Results

### 3.1 Cluster Structure

UMAP + HDBSCAN produced 20 clusters from 1,351 case-image pairs (362 noise points, 26.8%). Cluster sizes ranged from 20 to 120 cases.

### 3.2 Statistically Significant Phenotypes

After Bonferroni correction, 20 cluster × diagnosis combinations reached significance.

**Bipolar-Enriched MRI Clusters (Clusters 7 and 8)**

| Cluster | n | Diagnosis | % cluster | % rest | OR | p (corrected) |
|---------|---|-----------|-----------|--------|----|---------------|
| 8 | 109 | bipolar | 66.1% | 23.9% | 6.14 | <0.001 |
| 8 | 109 | depression | 13.8% | 42.3% | 0.22 | <0.001 |
| 7 | 39 | bipolar | 59.0% | 26.4% | 3.97 | 0.002 |
| 7 | 39 | depression | 10.3% | 40.9% | 0.18 | 0.026 |

**Depression-Dementia Co-occurrence Cluster (Cluster 9)**

| Cluster | n | Diagnosis | % cluster | % rest | OR | p (corrected) |
|---------|---|-----------|-----------|--------|----|---------------|
| 9 | 61 | depression | 70.5% | 38.5% | 3.75 | <0.001 |
| 9 | 61 | dementia | 24.6% | 2.2% | 14.25 | <0.001 |
| 9 | 61 | PTSD | 4.9% | 0.2% | 22.01 | 0.001 |
| 9 | 61 | bipolar | 4.9% | 28.4% | 0.15 | 0.012 |

**Severe Psychiatric Cluster (Cluster 17)**

| Cluster | n | Diagnosis | % cluster | % rest | OR | p (corrected) |
|---------|---|-----------|-----------|--------|----|---------------|
| 17 | 89 | psychosis | 34.8% | 6.8% | 7.32 | <0.001 |
| 17 | 89 | suicide | 23.6% | 6.3% | 4.67 | <0.001 |
| 17 | 89 | schizophrenia | 12.4% | 3.4% | 4.11 | 0.011 |

**Alzheimer-Anxiety Cluster (Cluster 0)**

| Cluster | n | Diagnosis | % cluster | % rest | OR | p (corrected) |
|---------|---|-----------|-----------|--------|----|---------------|
| 0 | 20 | alzheimer | 45.0% | 1.5% | 52.85 | <0.001 |

### 3.3 Ablation: Signal Source

| Condition | Clusters | Bipolar enrichment |
|-----------|----------|--------------------|
| Image only | 2 | None detected |
| Text only | 21 | Cluster: 84% bipolar, p<0.0001 |
| Combined | 18 | 59-66% bipolar, p<0.0001 |

Image-only clustering produced only 2 undifferentiated clusters with no diagnostic enrichment. Text-only clustering recovered strong bipolar separation. **The phenotypic signal is text-driven.**

### 3.4 External Validation on CNP

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | -0.027 | Random alignment |
| NMI | 0.062 | Minimal mutual information |
| Balanced accuracy | 0.250 | Exactly chance level (4 classes) |

Raw MRI embeddings did not recover diagnostic groupings, confirming that CLIP image features carry no psychiatric diagnostic signal.

---

## 4. Discussion

We present an unsupervised clustering analysis of 1,351 psychiatric case reports that produces statistically significant, clinically coherent phenotypes. However, our ablation and external validation establish clearly that these phenotypes are **narrative-driven, not imaging-driven**.

### 4.1 Clinical Narrative as a Phenotyping Signal

The text-only ablation recovered an 84% bipolar-enriched cluster (p<0.0001) that was absent in image-only clustering. This demonstrates that clinical case report text encodes strong diagnostic signal. Psychiatrists describe bipolar disorder cases using systematically different language than depression cases — different symptom profiles, treatment histories, clinical courses — and CLIP's text encoder captures this structure.

This finding has direct implications for clinical NLP. Large language models applied to psychiatric clinical notes, discharge summaries, or case reports may be a productive path toward computational psychiatric phenotyping, without requiring neuroimaging data at all.

### 4.2 Limitations of General-Purpose Image Embeddings for Psychiatry

CLIP was trained on natural image-text pairs from the internet. It does not encode the subtle structural brain differences — cortical thickness variations, white matter integrity, subcortical volume changes — that distinguish psychiatric conditions. Our CNP validation confirms this: chance-level classification accuracy on 265 subjects with confirmed diagnoses.

This is not a finding that psychiatric neuroimaging signal does not exist. The literature clearly demonstrates that it does. Rather, it is a finding that **CLIP is the wrong tool for psychiatric MRI**.

### 4.3 Future Work: Brain-Specific Models

For imaging signal to contribute to psychiatric phenotyping, future work should use:

1. **FreeSurfer-derived features** — cortical thickness, subcortical volumes, surface area. These have established validity in psychiatric neuroimaging and are interpretable by clinicians.

2. **Voxel-based morphometry (VBM)** — whole-brain voxel-level grey matter density maps, widely used in psychiatric research.

3. **Brain foundation models** — recently developed models pretrained specifically on neuroimaging data, such as BrainBERT or similar architectures, which encode brain-specific structure rather than natural image statistics.

4. **Larger datasets** — CNP (n=265) is underpowered for subtle psychiatric subtyping. UK Biobank (n=40,000+ with neuroimaging) would provide the statistical power needed to detect reliable subtypes.

### 4.4 Limitations

1. **Dataset bias:** MultiCaRe case reports represent unusual clinical presentations, not routine psychiatric practice.
2. **Sample size:** 1,351 Phase 1 cases; individual clusters as small as 20.
3. **NER limitations:** scispacy clinical NER is imperfect for complex clinical language.
4. **Imaging model:** The negative imaging result reflects CLIP's limitations, not the absence of imaging signal in psychiatric disorders.
5. **No prospective validation:** All findings are retrospective and hypothesis-generating.

---

## 5. Conclusion

Unsupervised clustering of multimodal psychiatric case reports produces clinically coherent phenotypes with statistically significant diagnosis enrichment. Ablation experiments and external validation demonstrate that this signal originates from clinical narrative text rather than brain imaging features. These findings establish the value of case report text for narrative-driven psychiatric phenotyping, while highlighting the need for brain-specific neuroimaging models to unlock imaging-driven psychiatric subtyping. The complete pipeline, code, and results are openly available at https://github.com/AliBaig-xD/multicare-psych.

---

## Acknowledgements

The author used Claude (Anthropic) as an AI research assistant throughout this work, for experimental design, code generation, debugging, and manuscript preparation. All scientific decisions, interpretations, and responsibility for the work remain with the author.

Data: MultiCaRe v2.0 (Nievas Offidani & Delrieux, 2023, Zenodo). CNP dataset: OpenNeuro ds000030.

---

## References

1. Nievas Offidani M, Delrieux C. The MultiCaRe Dataset: A Multimodal Case Report Dataset with Clinical Cases, Labeled Images and Captions from Open Access PMC Articles. Zenodo. 2023.

2. Poldrack RA, et al. A phenome-wide examination of neural and cognitive function. Scientific Data. 2016;3:160110.

3. Radford A, et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.

4. McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection. arXiv:1802.03426. 2018.

5. Campello RJGB, Moulavi D, Sander J. Density-Based Clustering Based on Hierarchical Density Estimates. PAKDD 2013.

6. Neumann M, et al. ScispaCy: Fast and Robust Models for Biomedical NLP. BioNLP 2019.

7. Drysdale AT, et al. Resting-state connectivity biomarkers define neurophysiological subtypes of depression. Nature Medicine. 2017;23:28-38.

8. Clementz BA, et al. Identification of Distinct Psychosis Biotypes Using Brain-Based Biomarkers. American Journal of Psychiatry. 2016;173:373-384.
