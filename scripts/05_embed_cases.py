"""
05_embed_cases.py
Phase C: Generate CLIP image + text embeddings for every row in the master table.

Each row produces:
  - A 512-dim image embedding  (from the image file)
  - A 512-dim text embedding   (from case_text[:1000] + image_caption[:500])

These are L2-normalised and concatenated → a 1024-dim vector per row.

Output:
  data/psych_brain_embeddings.parquet

Runtime: expect 1–4 hours on CPU depending on subset size.
         Run inside tmux so SSH disconnect won't kill it.

Run:
    tmux new -s embed        # or attach: tmux attach -t multicare
    source env/py/bin/activate
    python scripts/05_embed_cases.py | tee logs/05_embeddings.log
"""

import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

BASE_DIR     = "medical_datasets/psych_brain_multimodal"
MASTER_PATH  = "data/psych_brain_master.parquet"
OUT_PATH     = "data/psych_brain_embeddings.parquet"
BATCH_SIZE   = 16
CLIP_MODEL   = "openai/clip-vit-base-patch32"


def load_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def main():
    os.makedirs("data", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading CLIP model: {CLIP_MODEL} ...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.to(device)
    model.eval()

    df = pd.read_parquet(MASTER_PATH)
    print(f"Master table: {len(df)} rows")

    image_embeds = []
    text_embeds  = []
    keep_indices = []

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding batches"):
        end          = min(start + BATCH_SIZE, len(df))
        batch_df     = df.iloc[start:end]
        batch_images = []
        batch_texts  = []
        batch_idx    = []

        for idx, row in batch_df.iterrows():
            img_path = str(row["image_path"])  # already a full relative path from repo root
            img = load_image(img_path)
            if img is None:
                # Skip rows where the image file is missing / unreadable
                continue

            batch_images.append(img)
            case_text = str(row.get("case_text", ""))
            caption   = str(row.get("caption", ""))
            batch_texts.append(case_text[:1000] + " " + caption[:500])
            batch_idx.append(idx)

        if not batch_images:
            continue

        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,          # CLIP text encoder max tokens
        ).to(device)

        with torch.no_grad():
            outputs  = model(**inputs)
            img_feat = outputs.vision_model_output.pooler_output   # (B, 512)
            txt_feat = outputs.text_model_output.pooler_output     # (B, 512)

        # L2 normalise each modality separately
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)

        image_embeds.append(img_feat.cpu())
        text_embeds.append(txt_feat.cpu())
        keep_indices.extend(batch_idx)

    if not image_embeds:
        raise RuntimeError(
            "No embeddings were generated. "
            "Check that image paths in the master table resolve correctly."
        )

    image_embeds = torch.cat(image_embeds, dim=0)   # (N, 512)
    text_embeds  = torch.cat(text_embeds,  dim=0)   # (N, 512)
    combined     = torch.cat([image_embeds, text_embeds], dim=1)  # (N, 1024)

    emb_df = pd.DataFrame(combined.numpy())
    emb_df.insert(0, "orig_index", keep_indices)

    emb_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(emb_df)} embeddings ({combined.shape[1]}-dim) to {OUT_PATH}")
    skipped = len(df) - len(emb_df)
    if skipped:
        print(f"Skipped {skipped} rows (unreadable or missing images).")
    print("Next step: run scripts/06_cluster.py")


if __name__ == "__main__":
    main()
