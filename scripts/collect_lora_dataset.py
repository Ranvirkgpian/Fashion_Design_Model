import os, pandas as pd, shutil
from PIL import Image

ROOT     = r"E:\Fashion model"
IMG_DIR  = os.path.join(ROOT, r"data\deepfashion\raw\images")
CSV      = os.path.join(ROOT, r"data\deepfashion\raw\styles.csv")
LORA_DIR = os.path.join(ROOT, r"data\lora_dataset\minimalist_streetwear")
os.makedirs(LORA_DIR, exist_ok=True)

print("=" * 60)
print("  Week 4 Task 1 — Brand Style Dataset Collection")
print("  Style: Minimalist / Luxury Streetwear")
print("=" * 60)

df = pd.read_csv(CSV, on_bad_lines="skip")
df["id"] = df["id"].astype(str)

# ── Filter for minimalist streetwear aesthetics ────────────
mask = (
    df["masterCategory"].isin(["Apparel"]) &
    df["subCategory"].isin(["Topwear", "Bottomwear"]) &
    df["articleType"].isin([
        "Tshirts", "Shirts", "Jeans", "Trousers",
        "Casual Shoes", "Jackets", "Sweatshirts",
        "Track Pants", "Shorts"
    ]) &
    df["baseColour"].isin([
        "Black", "White", "Grey", "Navy Blue",
        "Off White", "Beige", "Charcoal"
    ]) &
    df["usage"].isin(["Casual"])
)

filtered = df[mask].copy()
print(f"\n  Matching items   : {len(filtered):,}")

# ── Filter to only existing images ────────────────────────
filtered["img_path"] = filtered["id"].apply(
    lambda x: os.path.join(IMG_DIR, f"{x}.jpg")
)
filtered = filtered[filtered["img_path"].apply(os.path.exists)]
print(f"  With images      : {len(filtered):,}")

# ── Sample 40 for LoRA training ───────────────────────────
sample = filtered.sample(min(40, len(filtered)), random_state=42).reset_index(drop=True)
print(f"  Selected for LoRA: {len(sample)}")

# ── Copy + resize to 512x512 ──────────────────────────────
print(f"\n  Processing images...")
captions = []
for i, row in sample.iterrows():
    dst = os.path.join(LORA_DIR, f"{i:03d}_{row['articleType'].lower().replace(' ','_')}.jpg")
    img = Image.open(row["img_path"]).convert("RGB").resize((512, 512))
    img.save(dst, quality=95)

    caption = (
        f"lux_street style, {row['baseColour'].lower()} "
        f"{row['articleType'].lower()}, minimalist luxury streetwear, "
        f"clean lines, high quality fashion"
    )
    txt_path = dst.replace(".jpg", ".txt")
    with open(txt_path, "w") as f:
        f.write(caption)
    captions.append({"file": os.path.basename(dst), "caption": caption})

    if (i+1) % 10 == 0:
        print(f"  Processed {i+1}/{len(sample)} images...")

# ── Save metadata ─────────────────────────────────────────
import json
meta_path = os.path.join(LORA_DIR, "dataset_metadata.json")
with open(meta_path, "w") as f:
    json.dump(captions, f, indent=2)

print(f"\n  Sample captions:")
print("─" * 60)
for c in captions[:5]:
    print(f"  [{c['file']}]")
    print(f"   {c['caption']}")
    print()

print("=" * 60)
print(f"  ✓  {len(sample)} images saved to: {LORA_DIR}")
print(f"  ✓  Captions written for each image")
print(f"  ✓  dataset_metadata.json saved")
print(f"  Ready for LoRA training!")
print("=" * 60)
