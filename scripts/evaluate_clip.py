import os, torch, warnings
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json

print("=" * 60)
print("  Week 2 Task 3 — CLIP Score Evaluation")
print("=" * 60)

# Load CLIP model
print("\n  Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("  ✓ CLIP model loaded")

# Images and their prompts
evaluations = [
    {
        "image": r"E:\Fashion model\outputs\01_pink_floral_dress.png",
        "prompt": "a woman wearing a pink floral summer dress fashion photography",
        "label": "Pink Floral Dress"
    },
    {
        "image": r"E:\Fashion model\outputs\02_navy_kurta_men.png",
        "prompt": "a man wearing a navy blue kurta traditional indian ethnic wear",
        "label": "Navy Blue Kurta"
    },
    {
        "image": r"E:\Fashion model\outputs\03_charcoal_suit.png",
        "prompt": "a man wearing a charcoal grey formal business suit white shirt",
        "label": "Charcoal Grey Suit"
    },
    {
        "image": r"E:\Fashion model\outputs\04_streetwear_women.png",
        "prompt": "a woman wearing white tshirt black jeans streetwear casual style",
        "label": "Streetwear Outfit"
    },
]

print("\n  Evaluating images...")
print("─" * 60)

results = []
for ev in evaluations:
    image = Image.open(ev["image"]).convert("RGB")
    inputs = processor(
        text=[ev["prompt"]],
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits_per_image.item()

    # Normalize to 0-100 scale
    normalized = min(100, max(0, (score / 30) * 100))

    results.append({
        "label": ev["label"],
        "raw_score": round(score, 2),
        "normalized": round(normalized, 1)
    })

    bar = "█" * int(normalized / 5)
    print(f"\n  {ev['label']}")
    print(f"  CLIP Score : {score:.2f}  ({normalized:.1f}/100)")
    print(f"  Quality    : {bar}")

# Summary
print("\n" + "=" * 60)
print("  CLIP Score Summary")
print("=" * 60)
avg = sum(r["raw_score"] for r in results) / len(results)
best = max(results, key=lambda x: x["raw_score"])
worst = min(results, key=lambda x: x["raw_score"])

print(f"\n  Average CLIP Score : {avg:.2f}")
print(f"  Best image         : {best['label']} ({best['raw_score']})")
print(f"  Needs improvement  : {worst['label']} ({worst['raw_score']})")

print("""
  Score Guide:
  > 25   Excellent — prompt and image strongly aligned
  20-25  Good — clear visual match
  15-20  Fair — some alignment issues
  < 15   Poor — prompt and image don't match
""")

# Save results
out_path = r"E:\Fashion model\outputs\clip_scores.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  ✓ Results saved to: {out_path}")
print("=" * 60)
