# Unsupervised Multimodal Clustering of Psychiatric Brain Imaging Case Reports Reveals Clinically Coherent Phenotypes Driven by Clinical Narrative

**Ali Baig**  
Independent Researcher  
Contact: [your email]

*Preprint — not peer reviewed*  
*April 2026*

---

## Abstract

Psychiatric diagnosis remains largely subjective, relying on clinical interviews rather than objective biomarkers. We applied unsupervised multimodal clustering to 1,351 brain-related psychiatric case reports from the MultiCaRe dataset — a large open-access collection of PubMed Central clinical case reports containing both images and clinical text. Each case was embedded using CLIP (Contrastive Language-Image Pretraining) to produce a combined image-text representation, which was then reduced with UMAP and clustered with HDBSCAN. Psychiatric diagnoses were extracted using clinical NER with negation handling (scispacy + negspacy) and cluster enrichment was validated with chi-square tests and Bonferroni correction. We identified 20 clusters, of which several showed statistically significant diagnosis enrichment: a bipolar-MRI phenotype (OR=6.14, p<0.001), a depression-dementia functional imaging cluster (OR=14.25 for dementia, p<0.001), and a severe psychosis-suicide cluster (OR=7.32, p<0.001). To determine the source of this signal, we performed an ablation study separating image and text embeddings. The bipolar separation was absent in image-only clustering but strongly present in text-only clustering (84% bipolar enrichment, p<0.0001), indicating that the signal originates from clinical narrative rather than imaging features. Validation on the Consortium for Neuropsychiatric Phenomics (CNP) dataset using raw T1w MRI embeddings confirmed this: ARI=-0.027, classification accuracy=0.250 (chance level). These findings suggest that multimodal clinical case reports are a rich resource for psychiatric phenotyping, but that current general-purpose image embeddings do not capture diagnostic imaging signal. Specialized neuroimaging models and larger datasets are needed to recover imaging-driven psychiatric subtypes.

---

## 1. Introduction

Psychiatric disorders — including bipolar disorder, schizophrenia, major depressive disorder, and anxiety — affect over 1 billion people worldwide and represent a leading cause of disability. Despite decades of neuroimaging research, no reliable imaging biomarkers have been established for routine clinical use. Diagnosis relies on clinical interviews, symptom checklists, and clinician judgment, leading to high rates of misdiagnosis and delayed treatment.

Unsupervised machine learning applied to neuroimaging data has shown promise in identifying psychiatric subtypes that cut across traditional diagnostic boundaries. Prior work has identified biotypes of depression using resting-state fMRI, subgroups of schizophrenia using structural MRI, and imaging-defined subtypes of Alzheimer's disease. However, these studies typically require large, carefully curated datasets with confirmed diagnoses, standardized imaging protocols, and institutional data access.

The MultiCaRe dataset offers an alternative: a large open-access collection of over 93,000 clinical case reports from PubMed Central, each containing clinical narrative text, associated images, image captions, and metadata. While case reports are not representative of the general clinical population, they represent a rich and underutilised resource for hypothesis generation and methodology development.

In this work, we present the first unsupervised multimodal clustering analysis of psychiatric brain imaging case reports from MultiCaRe. We combine image and text embeddings, apply density-based clustering, extract confirmed psychiatric diagnoses using clinical NLP, and statistically validate cluster enrichment. We then perform ablation experiments to determine the relative contribution of imaging versus narrative signal, and validate our approach on an independent dataset with confirmed DSM diagnoses.

---

## 2. Methods

### 2.1 Dataset

We used MultiCaRe v2.0, downloaded via the official `multiversity` Python toolkit. The full dataset contains 93,000+ clinical cases and 130,000+ images derived from open-access PubMed Central articles.

We filtered to a psychiatric brain imaging subset using the following criteria:
- Adult cases (age ≥ 18)
- Case text OR image caption contains at least one psychiatric term (schizophrenia, bipolar, mania, depression, psychosis, anxiety, PTSD, OCD, suicidal, self-harm)
- Image label indicates brain or head imaging (MRI, CT, head, brain, neuro, radiology)

This yielded 5,449 psychiatric cases. We then joined against the full image metadata (captions_and_labels.csv) filtered to head/brain radiology images, producing 1,515 candidate case-image pairs. After resolving file paths and removing rows with missing text or images, our final dataset contained **1,351 case-image pairs** across **4,901 unique cases**.

Imaging modalities: MRI (n=846, 62.6%), CT (n=378, 28.0%), PET (n=57, 4.2%), SPECT (n=38, 2.8%), X-ray (n=32, 2.4%).

### 2.2 Multimodal Embedding

Each case-image pair was embedded using CLIP ViT-B/32 (openai/clip-vit-base-patch32). For each pair:

- **Image embedding:** The image file was loaded and processed through the CLIP vision encoder, producing a 512-dimensional L2-normalised vector.
- **Text embedding:** The first 1,000 characters of case text were concatenated with the first 500 characters of the image caption and processed through the CLIP text encoder, producing a 512-dimensional L2-normalised vector.
- **Combined embedding:** Image and text vectors were concatenated to produce a 1,024-dimensional representation.

### 2.3 Clustering

Combined embeddings were standardised using StandardScaler, then reduced to 2 dimensions using UMAP (n_neighbors=15, min_dist=0.05, metric=cosine, random_state=42). Density-based clustering was performed using HDBSCAN (min_cluster_size=20, min_samples=5), producing 20 clusters with 26.8% noise points.

### 2.4 Diagnosis Extraction

Psychiatric diagnoses were extracted from case text using scispacy (en_core_sci_sm) with negspacy for negation detection. Entities were classified into 10 diagnostic categories: depression, bipolar, schizophrenia, psychosis, anxiety, PTSD, OCD, suicide/suicidality, dementia, and Alzheimer's disease.

Negated mentions ("no history of depression"), family history ("family history of bipolar"), and rule-out language ("rule out schizophrenia") were excluded. Only confirmed patient diagnoses were counted.

### 2.5 Statistical Validation

For each cluster × diagnosis combination, we computed:
- Chi-square test comparing diagnosis prevalence in the cluster vs the rest of the dataset
- Odds ratio with continuity correction
- Bonferroni correction for multiple testing (n_tests=200)

Significance threshold: p_corrected < 0.05.

### 2.6 Ablation Study

To determine the relative contribution of imaging versus narrative signal, we re-ran clustering separately on:
- **Image-only embeddings** (first 512 dimensions)
- **Text-only embeddings** (last 512 dimensions)
- **Combined embeddings** (all 1,024 dimensions)

We compared bipolar cluster enrichment across conditions.

### 2.7 External Validation (CNP Dataset)

We validated our approach on the Consortium for Neuropsychiatric Phenomics (CNP) dataset (OpenNeuro ds000030), which contains 265 subjects with confirmed DSM diagnoses: 125 healthy controls, 50 schizophrenia, 49 bipolar disorder, 41 ADHD.

For each subject, we downloaded the T1w structural MRI scan, extracted axial/sagittal/coronal slices, and embedded them using CLIP. We additionally extracted a 48-dimensional ROI-based brain fingerprint using the Harvard-Oxford cortical atlas via nilearn. Combined embeddings (560-dimensional) were clustered using the same UMAP + HDBSCAN pipeline.

Cluster-diagnosis alignment was measured using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI). Diagnostic separability was assessed using 5-fold cross-validated logistic regression balanced accuracy.

---

## 3. Results

### 3.1 Cluster Structure

UMAP + HDBSCAN produced 20 clusters from 1,351 case-image pairs, with 362 noise points (26.8%). Cluster sizes ranged from 20 to 120 cases. All clusters were exclusively head/brain radiology cases by design.

### 3.2 Statistically Significant Findings

After Bonferroni correction, 20 cluster × diagnosis combinations reached significance. Key findings:

**Bipolar MRI Phenotype (Clusters 7 and 8)**

Two large clusters showed strong bipolar enrichment with simultaneous depression depletion, both predominantly structural MRI:

| Cluster | n | Diagnosis | % cluster | % rest | OR | p (corrected) |
|---------|---|-----------|-----------|--------|----|---------------|
| 8 | 109 | bipolar | 66.1% | 23.9% | 6.14 | <0.001 |
| 8 | 109 | depression | 13.8% | 42.3% | 0.22 | <0.001 |
| 7 | 39 | bipolar | 59.0% | 26.4% | 3.97 | 0.002 |
| 7 | 39 | depression | 10.3% | 40.9% | 0.18 | 0.026 |

**Depression-Dementia Functional Imaging Cluster (Cluster 9)**

A cluster dominated by PET and SPECT functional imaging showed co-enrichment of depression and dementia diagnoses:

| Cluster | n | Diagnosis | % cluster | % rest | OR | p (corrected) |
|---------|---|-----------|-----------|--------|----|---------------|
| 9 | 61 | depression | 70.5% | 38.5% | 3.75 | <0.001 |
| 9 | 61 | dementia | 24.6% | 2.2% | 14.25 | <0.001 |
| 9 | 61 | PTSD | 4.9% | 0.2% | 22.01 | 0.001 |
| 9 | 61 | bipolar | 4.9% | 28.4% | 0.15 | 0.012 |

**Severe Psychiatric MRI Cluster (Cluster 17)**

| Cluster | n | Diagnosis | % cluster | % rest | OR | p (corrected) |
|---------|---|-----------|-----------|--------|----|---------------|
| 17 | 89 | psychosis | 34.8% | 6.8% | 7.32 | <0.001 |
| 17 | 89 | suicide | 23.6% | 6.3% | 4.67 | <0.001 |
| 17 | 89 | schizophrenia | 12.4% | 3.4% | 4.11 | 0.011 |

**Alzheimer-Anxiety Cluster (Cluster 0)**

| Cluster | n | Diagnosis | % cluster | % rest | OR | p (corrected) |
|---------|---|-----------|-----------|--------|----|---------------|
| 0 | 20 | alzheimer | 45.0% | 1.5% | 52.85 | <0.001 |

### 3.3 Ablation: Image vs Text Signal

To determine the source of the observed clustering signal, we re-ran clustering on image-only, text-only, and combined embeddings separately.

| Condition | Clusters | Bipolar enrichment |
|-----------|----------|--------------------|
| Image only | 2 | None detected |
| Text only | 21 | Cluster 7: 84% bipolar, p<0.0001 |
| Combined | 18 | Multiple clusters: 59-66% bipolar, p<0.0001 |

Image-only clustering produced only 2 clusters with no diagnostic enrichment. Text-only clustering recovered strong bipolar separation (84% enrichment). Combined embeddings produced intermediate results. This indicates that the observed phenotypic signal originates predominantly from clinical narrative text rather than imaging features.

### 3.4 External Validation on CNP

Applying our pipeline to 265 CNP subjects with confirmed DSM diagnoses:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | -0.027 | Random alignment |
| NMI | 0.062 | Minimal mutual information |
| Balanced accuracy | 0.250 | Exactly at chance level |
| Chance level | 0.250 | (4 classes) |

Clustering on raw T1w MRI embeddings (CLIP slices + ROI fingerprint) did not recover diagnostic groupings. This is consistent with the ablation finding that imaging signal does not drive the MultiCaRe clusters.

---

## 4. Discussion

We present an unsupervised multimodal clustering analysis of 1,351 psychiatric brain imaging case reports, identifying statistically significant phenotypic clusters that survive Bonferroni correction. The strongest findings — bipolar disorder separation from depression, depression-dementia co-occurrence with functional imaging, and severe psychosis-suicide clustering — are clinically coherent and align with known patterns in the psychiatric neuroimaging literature.

However, our ablation analysis reveals a critical finding: the observed signal is driven by clinical narrative text, not brain imaging features. CLIP image embeddings alone produce no diagnostic clustering, while text embeddings alone recover the bipolar separation with high enrichment. This was confirmed by our external validation on CNP, where raw MRI embeddings produced chance-level classification accuracy.

This has two important implications:

**First**, clinical case report text is a rich signal for psychiatric phenotyping. The language used to describe bipolar disorder cases is systematically different from depression cases in ways that CLIP's text encoder captures. This suggests that large language models applied to psychiatric clinical notes could be a productive direction for computational psychiatry.

**Second**, general-purpose image embeddings (CLIP) are insufficient for psychiatric neuroimaging. CLIP was trained on natural images and does not encode the subtle structural brain differences — cortical thickness, white matter integrity, subcortical volumes — that distinguish psychiatric conditions. Specialized neuroimaging models such as those based on FreeSurfer-derived features, voxel-based morphometry, or brain-specific foundation models would be needed to recover imaging-driven psychiatric signal.

### Limitations

1. **Dataset bias:** MultiCaRe case reports represent unusual or instructive clinical presentations, not routine psychiatric practice. Findings may not generalise to the broader clinical population.

2. **Sample size:** 1,351 cases total. Individual clusters as small as 20 cases have limited statistical power.

3. **NER limitations:** scispacy clinical NER is imperfect. Complex clinical language and implicit diagnoses may be misclassified.

4. **Imaging model:** CLIP was not designed for medical imaging. The negative imaging result reflects this limitation rather than the absence of imaging signal in psychiatric disorders.

5. **No prospective validation:** All findings are retrospective and hypothesis-generating.

---

## 5. Conclusion

Unsupervised multimodal clustering of psychiatric case reports produces clinically coherent phenotypic clusters with statistically significant diagnosis enrichment. However, ablation experiments and external validation demonstrate that this signal is driven by clinical narrative text rather than brain imaging features. These findings highlight both the potential of clinical text for psychiatric phenotyping and the need for specialized neuroimaging models to extract diagnostic signal from brain scans. The complete pipeline, code, and results are openly available at https://github.com/AliBaig-xD/multicare-psych.

---

## Acknowledgements

The author used Claude (Anthropic) as an AI research assistant throughout this work, for experimental design, code generation, debugging, and manuscript preparation. All scientific decisions, interpretations, and responsibility for the work remain with the author.

Data: MultiCaRe v2.0 (Nievas Offidani & Delrieux, 2023, Zenodo). CNP dataset: OpenNeuro ds000030.

---

## References

1. Nievas Offidani M, Delrieux C. The MultiCaRe Dataset: A Multimodal Case Report Dataset with Clinical Cases, Labeled Images and Captions from Open Access PMC Articles. Zenodo. 2023. https://doi.org/10.5281/zenodo.10079370

2. Poldrack RA, et al. A phenome-wide examination of neural and cognitive function. Scientific Data. 2016;3:160110. (CNP dataset)

3. Radford A, et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021. (CLIP)

4. McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426. 2018.

5. Campello RJGB, Moulavi D, Sander J. Density-Based Clustering Based on Hierarchical Density Estimates. PAKDD 2013. (HDBSCAN)

6. Neumann M, et al. ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing. BioNLP 2019.

7. Drysdale AT, et al. Resting-state connectivity biomarkers define neurophysiological subtypes of depression. Nature Medicine. 2017;23:28-38.

8. Clementz BA, et al. Identification of Distinct Psychosis Biotypes Using Brain-Based Biomarkers. American Journal of Psychiatry. 2016;173:373-384.