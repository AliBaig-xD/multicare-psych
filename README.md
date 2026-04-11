# MultiCaRe Psychiatric Imaging Atlas

**Author:** Ali Baig (Independent Researcher)

Unsupervised discovery of latent psychiatric brain-imaging phenotypes from the
[MultiCaRe](https://github.com/gcapde/MultiCaRe) multimodal clinical case dataset.

---

## What this does

This project builds an unsupervised "atlas" of psychiatric brain imaging patterns by:

1. Downloading MultiCaRe v2.0 — ~93k clinical cases and ~130k images from PubMed Central
2. Filtering to adult psychiatric cases with brain/head imaging (1,351 cases)
3. Embedding each case (brain image + clinical text) using CLIP
4. Clustering embeddings with UMAP + HDBSCAN
5. Extracting confirmed psychiatric diagnoses using scispacy NER with negation handling
6. Validating cluster enrichment with chi-square tests and Bonferroni correction
7. Serving an interactive atlas via Streamlit

---

## Key Findings (Phase 1)

Running unsupervised clustering on 1,351 psychiatric brain imaging cases produced
20 clusters. Statistical analysis (chi-square, Bonferroni corrected) revealed
several clusters with significant diagnosis enrichment:

### Cluster 8 & 7 — Bipolar MRI Phenotype (strongest finding)
- **66% bipolar** enrichment (OR=6.14, p<0.001)
- Simultaneously **depleted for depression** (OR=0.22, p<0.001)
- Pure structural MRI (79/109 cases MRI)
- The algorithm separated bipolar from depression cases on brain MRI without supervision

### Cluster 9 — Depression-Dementia Functional Imaging
- **70% depression** (OR=3.75, p<0.001)
- **25% dementia** (OR=14.25, p<0.001)
- PET/SPECT dominant — functional metabolic imaging
- Clinically coherent: PET is used to distinguish depression from early dementia

### Cluster 17 — Severe Psychiatric MRI Phenotype
- **Psychosis 35%** (OR=7.32, p<0.001)
- **Suicide risk 24%** (OR=4.67, p<0.001)
- **Schizophrenia 12%** (OR=4.11, p<0.05)
- Pure MRI — severe/complex psychiatric presentations

### Cluster 0 — Alzheimer-Anxiety
- **Alzheimer 45%** (OR=52.85, p<0.001) — highest odds ratio in the dataset

### Cluster 2 & 19 — Psychosis/Schizophrenia MRI
- Schizophrenia OR=9.27 (p<0.001), Psychosis OR=5.49 (p<0.001)

Full results: [`results/significant_findings.csv`](results/significant_findings.csv)

---

## Repo Structure

```
multicare-psych/
├── scripts/
│   ├── 01_download_multicare.py      # Download dataset
│   ├── 02_build_psych_subset.py      # Filter to psych/brain cases
│   ├── 03_inspect_schema.py          # Inspect output schema
│   ├── 04_build_master_table.py      # Build unified parquet table
│   ├── 05_embed_cases.py             # CLIP embeddings
│   ├── 06_cluster.py                 # UMAP + HDBSCAN clustering
│   ├── 07_extract_diagnoses.py       # NER diagnosis extraction
│   └── 08_statistical_analysis.py   # Chi-square + Bonferroni
├── app/
│   └── atlas.py                      # Streamlit interactive atlas
├── results/
│   ├── significant_findings.csv      # Bonferroni-corrected results
│   └── statistical_summary.csv       # Full stats
├── logs/                             # Pipeline logs
├── requirements.txt
├── setup.sh
├── FINDINGS.md
└── README.md
```

---

## How to Reproduce

### 1. Provision a VPS
Ubuntu 22.04, 4 vCPU, 16 GB RAM, 150 GB SSD (AWS t3.xlarge or equivalent)

### 2. Clone and set up
```bash
git clone git@github.com:AliBaig-xD/multicare-psych.git
cd multicare-psych
bash setup.sh
source env/py/bin/activate
pip install negspacy
```

### 3. Run the pipeline
```bash
tmux new -s multicare
source env/py/bin/activate

python scripts/01_download_multicare.py   | tee logs/01_download.log
python scripts/02_build_psych_subset.py   | tee logs/02_psych_subset.log
python scripts/04_build_master_table.py   | tee logs/04_master_table.log
python scripts/05_embed_cases.py          | tee logs/05_embeddings.log
python scripts/06_cluster.py             | tee logs/06_clusters.log
python scripts/07_extract_diagnoses.py    | tee logs/07_diagnoses.log
python scripts/08_statistical_analysis.py | tee logs/08_stats.log
```

### 4. Explore the atlas
```bash
streamlit run app/atlas.py --server.port 8501 --server.address 0.0.0.0
```

SSH tunnel from your Mac:
```bash
ssh -L 8501:localhost:8501 ubuntu@<your_vps_ip>
```
Then open: http://localhost:8501

---

## Caveats

- Dataset bias: case reports are not representative medicine
- Sample size: 1,351 cases is small for strong claims
- Modality confound partially present — MRI vs CT vs PET drives some clustering
- No external validation yet (Phase 2 planned)

---

## Roadmap

- [x] Phase 1: Proof of concept on MultiCaRe case reports
- [ ] Phase 2: Validation on CNP dataset (OpenNeuro ds000030)
- [ ] Phase 3: Preprint

---

## License
Code: MIT — Data: Subject to MultiCaRe dataset license (CC BY)