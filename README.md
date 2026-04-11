# MultiCaRe Psychiatric Imaging Atlas

Unsupervised discovery of latent psychiatric brain-imaging phenotypes from the
[MultiCaRe](https://github.com/gcapde/MultiCaRe) multimodal clinical case dataset.

---

## What this does

1. Downloads MultiCaRe v2.0 (~93k clinical cases, ~130k images from PubMed Central)
2. Filters to adult psychiatric cases with brain/head imaging
3. Embeds each case (image + text) with CLIP
4. Clusters embeddings with UMAP + HDBSCAN
5. Serves an interactive atlas via Streamlit

---

## Repo structure

```
multicare-psych/
├── scripts/
│   ├── 01_download_multicare.py   # Phase A: download dataset
│   ├── 02_build_psych_subset.py   # Phase A: filter to psych/brain cases
│   ├── 03_inspect_schema.py       # Phase A: inspect output before proceeding
│   ├── 04_build_master_table.py   # Phase B: build unified parquet table
│   ├── 05_embed_cases.py          # Phase C: CLIP embeddings
│   └── 06_cluster.py              # Phase D: UMAP + HDBSCAN
├── app/
│   └── atlas.py                   # Phase E: Streamlit atlas
├── data/                          # generated — gitignored
├── results/                       # generated — gitignored
├── logs/                          # generated — gitignored
├── medical_datasets/              # downloaded data — gitignored
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Step-by-step execution

### 0. Prerequisites

- Ubuntu 22.04 VPS (4 vCPU, 16 GB RAM, 150 GB SSD minimum)
- Git installed
- SSH access

### 1. Clone and set up

```bash
git clone https://github.com/<your-username>/multicare-psych.git
cd multicare-psych
bash setup.sh
source env/py/bin/activate
```

### 2. Start a tmux session (keeps processes alive on SSH disconnect)

```bash
tmux new -s multicare
source env/py/bin/activate
```

### 3. Phase A — Download & filter

```bash
python scripts/01_download_multicare.py | tee logs/01_download.log
python scripts/02_build_psych_subset.py | tee logs/02_psych_subset.log
python scripts/03_inspect_schema.py     | tee logs/03_schema.log
```

> ⚠️ **Stop here** after step 3. Open `logs/03_schema.log`, check the column
> names, and update the `COLUMN MAPPING` section in
> `scripts/04_build_master_table.py` before continuing.

### 4. Phase B — Master table

```bash
python scripts/04_build_master_table.py | tee logs/04_master_table.log
```

Verify `data/psych_brain_master.parquet` looks correct before continuing.

### 5. Phase C — Embeddings (long-running, keep in tmux)

```bash
python scripts/05_embed_cases.py | tee logs/05_embeddings.log
```

Expect 1–4 hours on CPU. Detach tmux with `Ctrl+B D`, reattach with
`tmux attach -t multicare`.

### 6. Phase D — Clustering

```bash
python scripts/06_cluster.py | tee logs/06_clusters.log
```

### 7. Phase E — Atlas

```bash
streamlit run app/atlas.py --server.port 8501 --server.address 0.0.0.0
```

**Access from your local machine** via SSH tunnel (no firewall changes needed):

```bash
ssh -L 8501:localhost:8501 ubuntu@<your_vps_ip>
```

Then open: http://localhost:8501

---

## Notes

- The `data/`, `results/`, `logs/`, and `medical_datasets/` directories are
  gitignored — only code is committed.
- CLIP model weights (~600 MB) are downloaded automatically on first run of
  `05_embed_cases.py` via HuggingFace Hub; they are cached in `~/.cache/`.
- For faster embedding, switch to a GPU instance and remove the `--index-url`
  CPU flag from `setup.sh`.
