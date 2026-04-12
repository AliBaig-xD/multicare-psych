"""
11_embed_cnp.py
Phase 2 – Step 3: Embed CNP brain scans using two complementary approaches:

  A) CLIP on 2D slices  — directly comparable to Phase 1
  B) 3D brain encoder   — uses full volumetric information via nilearn + a
                          pretrained ResNet feature extractor on NIfTI volumes

On a GPU instance both run fast. The final embedding concatenates both.

Output:
  cnp_data/cnp_embeddings.parquet

Run:
    source env/py/bin/activate
    python scripts/11_embed_cnp.py | tee logs/11_embed_cnp.log
"""

import os
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from nilearn.image import reorder_img, smooth_img
from nilearn.masking import compute_brain_mask

CNP_MASTER  = "cnp_data/cnp_master.parquet"
OUT_PARQUET = "cnp_data/cnp_embeddings.parquet"
BATCH_SIZE  = 32
CLIP_MODEL  = "openai/clip-vit-base-patch32"


# ── 3D volumetric feature extractor ──────────────────────────────────────────
class VolumetricEncoder:
    """
    Lightweight 3D brain encoder using nilearn signal extraction.
    Extracts mean signal from anatomical ROIs as a compact brain fingerprint.
    Fast on CPU/GPU, no large pretrained weights needed.
    """

    def __init__(self):
        from nilearn import datasets
        # Use Harvard-Oxford cortical atlas ROIs
        print("  Loading brain atlas...")
        self.atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        self.atlas_img = nib.load(self.atlas.maps)
        self.atlas_data = self.atlas_img.get_fdata()
        self.n_rois = int(self.atlas_data.max())
        print(f"  Atlas loaded: {self.n_rois} ROIs")

    def encode(self, nii_path: str) -> np.ndarray | None:
        """Extract per-ROI mean intensity → compact brain fingerprint vector."""
        try:
            img = nib.load(nii_path)
            img = reorder_img(img, resample="continuous")
            img = smooth_img(img, fwhm=6)    # light smoothing

            # Resample to atlas space
            from nilearn.image import resample_to_img
            img_resampled = resample_to_img(img, self.atlas_img, interpolation="linear")
            data = img_resampled.get_fdata()

            # Extract mean signal per ROI
            features = np.zeros(self.n_rois)
            for roi_idx in range(1, self.n_rois + 1):
                mask = self.atlas_data == roi_idx
                if mask.sum() > 0:
                    features[roi_idx - 1] = data[mask].mean()

            # L2 normalise
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features.astype(np.float32)

        except Exception as e:
            print(f"  Volumetric encoding failed for {nii_path}: {e}")
            return None


def load_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def embed_clip(df: pd.DataFrame, model, processor, device) -> dict:
    """Embed 2D slices with CLIP. Returns dict: sub_id -> mean embedding."""
    sub_embeddings = {}

    # Group by subject — average embeddings across 3 views
    for sub_id, group in tqdm(df.groupby("sub_id"), desc="CLIP embedding"):
        images = []
        for _, row in group.iterrows():
            img = load_image(row["slice_path"])
            if img:
                images.append(img)

        if not images:
            continue

        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            feats = model.get_image_features(**inputs)   # (n_views, 512)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            # Average across views
            mean_feat = feats.mean(dim=0)
            mean_feat = torch.nn.functional.normalize(mean_feat.unsqueeze(0), p=2).squeeze(0)

        sub_embeddings[sub_id] = mean_feat.cpu().numpy()

    return sub_embeddings


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = pd.read_parquet(CNP_MASTER)
    subjects = df.drop_duplicates("sub_id")[["sub_id", "nii_path", "diagnosis", "age", "gender"]]
    print(f"Subjects to embed: {len(subjects)}")

    # ── A) CLIP on 2D slices ──────────────────────────────────────────────────
    print("\n[1/2] CLIP embedding on 2D slices...")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model.eval()

    clip_embeds = embed_clip(df, clip_model, clip_processor, device)
    print(f"  CLIP embeddings: {len(clip_embeds)} subjects")

    # Free GPU memory before 3D step
    del clip_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── B) Volumetric 3D encoding ─────────────────────────────────────────────
    print("\n[2/2] Volumetric 3D encoding (ROI-based brain fingerprint)...")
    vol_encoder = VolumetricEncoder()
    vol_embeds  = {}

    for _, row in tqdm(subjects.iterrows(), total=len(subjects), desc="Volumetric"):
        feat = vol_encoder.encode(row["nii_path"])
        if feat is not None:
            vol_embeds[row["sub_id"]] = feat

    print(f"  Volumetric embeddings: {len(vol_embeds)} subjects")

    # ── Combine ───────────────────────────────────────────────────────────────
    rows = []
    for _, row in subjects.iterrows():
        sub_id = row["sub_id"]
        if sub_id not in clip_embeds or sub_id not in vol_embeds:
            continue

        clip_feat = clip_embeds[sub_id]          # 512-dim
        vol_feat  = vol_embeds[sub_id]           # n_rois-dim

        combined = np.concatenate([clip_feat, vol_feat])

        entry = {
            "sub_id":    sub_id,
            "diagnosis": row["diagnosis"],
            "age":       row["age"],
            "gender":    row["gender"],
        }
        for i, v in enumerate(combined):
            entry[f"feat_{i}"] = v

        rows.append(entry)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT_PARQUET, index=False)

    print(f"\nSaved {len(out_df)} subject embeddings to {OUT_PARQUET}")
    print(f"Embedding dim: {len(combined)}")
    print("\nDiagnosis breakdown:")
    print(out_df["diagnosis"].value_counts().to_string())
    print("\nNext step: run scripts/12_cluster_cnp.py")


if __name__ == "__main__":
    main()
