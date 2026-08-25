import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, torch, warnings
warnings.filterwarnings("ignore")
os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"

import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print("=" * 60)
print("  Week 5 Task 2 — CLIP Embeddings (CORRECTED projection)")
print("=" * 60)

ROOT = r"E:\Fashion model"
CSV  = os.path.join(ROOT, r"data\trend_database\trend_database.csv")
IMG_DIR = os.path.join(ROOT, r"data\deepfashion\raw\images")
OUT_DIR = os.path.join(ROOT, r"data\embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 2000

print(f"\n  Device: {DEVICE}")
print("  Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
model.eval()
print("  CLIP loaded")

df = pd.read_csv(CSV)
df["id"] = df["id"].astype(str)
df["img_path"] = df["id"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))
df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)

sample = df.groupby("trend_category", group_keys=False).apply(
    lambda x: x.sample(min(len(x), max(1, int(SAMPLE_SIZE * len(x) / len(df)))), random_state=42)
).reset_index(drop=True)
sample = sample.head(SAMPLE_SIZE)
print(f"  Sampled {len(sample):,} items")

# CORRECT manual projection — bypasses the broken get_image_features wrapper
def encode_images(images):
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        projected = model.visual_projection(vision_out.pooler_output)
        projected = projected / projected.norm(dim=-1, keepdim=True)
    return projected.cpu().numpy()

print("\n  Embedding images (corrected projection)...")
BATCH = 32
all_embeddings = []
all_ids = []

for i in range(0, len(sample), BATCH):
    batch = sample.iloc[i:i+BATCH]
    images = [Image.open(p).convert("RGB") for p in batch["img_path"]]
    emb = encode_images(images)
    all_embeddings.append(emb)
    all_ids.extend(batch["id"].tolist())
    if (i // BATCH) % 10 == 0:
        print(f"  Embedded {min(i+BATCH, len(sample))}/{len(sample)} images...")

embeddings = np.vstack(all_embeddings)
print(f"\n  Done! Shape: {embeddings.shape}")

np.save(os.path.join(OUT_DIR, "clip_embeddings.npy"), embeddings)
ordered_metadata = df.set_index("id").loc[all_ids].reset_index()
save_cols = ["id","articleType","baseColour","gender","trend_category","trend_score","search_description","img_path"]
ordered_metadata[save_cols].to_csv(os.path.join(OUT_DIR, "embedding_metadata.csv"), index=False)

# Quick sanity check: text-image similarity should now be sensible
def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        txt_out = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        projected = model.text_projection(txt_out.pooler_output)
        projected = projected / projected.norm(dim=-1, keepdim=True)
    return projected.cpu().numpy()

test_text_emb = encode_text("a black tshirt")
sims = embeddings @ test_text_emb.T
best_idx = sims.flatten().argsort()[::-1][:3]
print("\n  Sanity check — query: 'a black tshirt'")
for idx in best_idx:
    print(f"    [{sims[idx][0]:.3f}] {ordered_metadata.iloc[idx]['search_description']}")

print("\n" + "=" * 60)
print(f"  Corrected embeddings saved!")
print(f"  Shape: {embeddings.shape[0]:,} x {embeddings.shape[1]}")
print("=" * 60)
