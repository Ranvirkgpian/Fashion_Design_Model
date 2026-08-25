import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, torch, json, warnings
warnings.filterwarnings("ignore")
os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print("=" * 60)
print("  Week 8 Task 4 — Lookbook Style Analysis")
print("=" * 60)

LOOKBOOK_DIR = r"E:\Fashion model\outputs\lookbook"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
model.eval()

with open(os.path.join(LOOKBOOK_DIR, "lookbook_manifest.json")) as f:
    manifest = json.load(f)

collections = {}
for item in manifest:
    if item["status"] != "success":
        continue
    coll = item["collection"]
    collections.setdefault(coll, []).append(item["file"])

print(f"\n  Analyzing {len(manifest)} designs across {len(collections)} collections...\n")

collection_prompts = {
    "Evening Elegance": "elegant evening gown, formal luxury fashion",
    "Minimalist Streetwear": "minimalist streetwear, clean neutral tones",
    "Ethnic Heritage": "traditional indian ethnic wear, cultural fashion",
    "Business Formal": "professional business formal attire",
    "Casual Everyday": "casual everyday comfortable clothing",
    "Sport & Athleisure": "athletic sportswear, activewear",
}

results = {}
for coll, files in collections.items():
    text_query = collection_prompts.get(coll, coll)
    text_inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_out = model.text_model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
        text_emb = model.text_projection(text_out.pooler_output)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    scores = []
    for f in files:
        img_path = os.path.join(LOOKBOOK_DIR, f)
        image = Image.open(img_path).convert("RGB")
        img_inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=img_inputs["pixel_values"])
            img_emb = model.visual_projection(vision_out.pooler_output)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb @ text_emb.T).item()
        scores.append(sim)

    avg_score = sum(scores) / len(scores)
    results[coll] = {"files": files, "scores": [round(s,3) for s in scores], "avg_score": round(avg_score,3)}
    bar = "█" * int(avg_score * 60)
    print(f"  {coll:<24} avg: {avg_score:.3f}  {bar}")

overall_avg = sum(r["avg_score"] for r in results.values()) / len(results)

with open(os.path.join(LOOKBOOK_DIR, "style_analysis.json"), "w") as f:
    json.dump({"collections": results, "overall_avg": round(overall_avg,3), "total_designs": len(manifest)}, f, indent=2)

print()
print("=" * 60)
print(f"  Overall lookbook CLIP consistency: {overall_avg:.3f}")
print(f"  Total designs analyzed: {len(manifest)}")
print(f"  Report saved: style_analysis.json")
print("=" * 60)
