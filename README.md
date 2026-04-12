# MultiCaRe Psychiatric Imaging Atlas

**Author:** Ali Baig (Independent Researcher)  
**Preprint:** See [PREPRINT.md](PREPRINT.md)

Unsupervised discovery of latent psychiatric brain-imaging phenotypes from the
[MultiCaRe](https://github.com/gcapde/MultiCaRe) multimodal clinical case dataset,
with external validation on the CNP dataset (OpenNeuro ds000030).

---

## Key Finding

Unsupervised clustering of 1,351 psychiatric brain imaging case reports produces
clinically coherent phenotypes (bipolar MRI cluster OR=6.14 p<0.001, depression-dementia
PET cluster OR=14.25 p<0.001). However, ablation experiments show the signal is driven
by **clinical narrative text, not imaging features**. External validation on 265 CNP
subjects with confirmed DSM diagnoses confirms this: ARI=-0.027, accuracy=chance level.

---

## Repo Structure

```
multicare-psych/
├── scripts/
│   ├── 01_download_multicare.py      # Download MultiCaRe
│   ├── 02_build_psych_subset.py      # Filter to psych/brain cases
│   ├── 03_inspect_schema.py          # Inspect schema
│   ├── 04_build_master_table.py      # Build master parquet table
│   ├── 05_embed_cases.py             # CLIP embeddings
│   ├── 06_cluster.py                 # UMAP + HDBSCAN
│   ├── 07_extract_diagnoses.py       # NER diagnosis extraction
│   ├── 08_statistical_analysis.py   # Chi-square + Bonferroni
│   ├── 09_download_cnp.py            # Download CNP dataset
│   ├── 10_preprocess_cnp.py          # Extract NIfTI slices
│   ├── 11_embed_cnp.py               # Embed CNP scans
│   ├── 12_cluster_cnp.py             # Cluster CNP
│   └── 13_validate_cnp.py            # Validate against DSM labels
├── app/
│   └── atlas.py                      # Streamlit interactive atlas
├── results/
│   ├── significant_findings.csv      # Phase 1 Bonferroni results
│   ├── statistical_summary.csv       # Full Phase 1 stats
│   ├── cnp_cluster_summary.csv       # Phase 2 cluster breakdown
│   └── cnp_validation.csv            # Phase 2 validation metrics
├── logs/                             # All pipeline logs
├── PREPRINT.md                       # Full manuscript
├── FINDINGS.md                       # Summary of findings
├── requirements.txt
└── setup.sh
```

---

## How to Reproduce

### Phase 1 (MultiCaRe)

**Requirements:** Ubuntu 22.04, 4 vCPU, 16 GB RAM, 150 GB SSD

```bash
git clone git@github.com:AliBaig-xD/multicare-psych.git
cd multicare-psych
bash setup.sh
source env/py/bin/activate

python scripts/01_download_multicare.py   | tee logs/01_download.log
python scripts/02_build_psych_subset.py   | tee logs/02_psych_subset.log
python scripts/04_build_master_table.py   | tee logs/04_master_table.log
python scripts/05_embed_cases.py          | tee logs/05_embeddings.log
python scripts/06_cluster.py             | tee logs/06_clusters.log
python scripts/07_extract_diagnoses.py    | tee logs/07_diagnoses.log
python scripts/08_statistical_analysis.py | tee logs/08_stats.log
```

### Phase 2 (CNP Validation)

**Requirements:** Same + GPU recommended (NVIDIA T4 or better)

```bash
python scripts/09_download_cnp.py    | tee logs/09_cnp_download.log
python scripts/10_preprocess_cnp.py  | tee logs/10_preprocess.log
python scripts/11_embed_cnp.py       | tee logs/11_embed_cnp.log
python scripts/12_cluster_cnp.py     | tee logs/12_cluster_cnp.log
python scripts/13_validate_cnp.py    | tee logs/13_validation.log
```

### Atlas

```bash
streamlit run app/atlas.py --server.port 8501 --server.address 0.0.0.0
# SSH tunnel from Mac: ssh -L 8501:localhost:8501 ubuntu@<vps_ip>
# Open: http://localhost:8501
```

---

## Results

| Phase | Dataset | n | Key metric |
|-------|---------|---|------------|
| 1 | MultiCaRe | 1,351 | Bipolar OR=6.14, p<0.001 |
| 1 (ablation) | MultiCaRe text-only | 1,351 | Bipolar 84%, p<0.0001 |
| 1 (ablation) | MultiCaRe image-only | 1,351 | No signal |
| 2 | CNP ds000030 | 265 | ARI=-0.027, acc=chance |

---

## Acknowledgements

The author used Claude (Anthropic) as an AI research assistant throughout this
work, for experimental design, code generation, debugging, and manuscript
preparation.

---

## License
Code: MIT — Data: Subject to MultiCaRe (CC BY) and CNP dataset licenses
