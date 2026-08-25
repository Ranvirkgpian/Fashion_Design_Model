import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
model.eval()

# Load ONE real image from your dataset
ROOT = r"E:\Fashion model"
img_path = r"E:\Fashion model\data\deepfashion\raw\images\15970.jpg"

if not os.path.exists(img_path):
    # find any image that exists
    img_dir = os.path.join(ROOT, r"data\deepfashion\raw\images")
    files = os.listdir(img_dir)[:1]
    img_path = os.path.join(img_dir, files[0])

print("Using image:", img_path)
image = Image.open(img_path).convert("RGB")

# Method used in embed_clip.py (Task 2)
img_inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    img_emb_v1 = model.get_image_features(**img_inputs)
    img_emb_v1 = img_emb_v1 / img_emb_v1.norm(dim=-1, keepdim=True)
print("\nMethod 1 (get_image_features) - shape:", img_emb_v1.shape)
print("First 5 values:", img_emb_v1[0][:5].cpu().numpy())

# Text encoding using same get_text_features-style approach
text_inputs = processor(text=["a photo of a shirt"], return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    txt_out = model.text_model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
    txt_emb = model.text_projection(txt_out.pooler_output)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
print("\nText embedding shape:", txt_emb.shape)
print("First 5 values:", txt_emb[0][:5].cpu().numpy())

# Compute cosine similarity directly
sim = (img_emb_v1 @ txt_emb.T).item()
print(f"\nDirect cosine similarity (image vs text): {sim:.4f}")

# Also check what get_image_features actually does internally
print("\n--- Checking get_image_features implementation ---")
with torch.no_grad():
    vision_out = model.vision_model(pixel_values=img_inputs["pixel_values"])
    pooled_img = vision_out.pooler_output
    projected_img = model.visual_projection(pooled_img)
    projected_img = projected_img / projected_img.norm(dim=-1, keepdim=True)
print("Manual projection first 5 values:", projected_img[0][:5].cpu().numpy())
print("Matches get_image_features?", torch.allclose(img_emb_v1, projected_img, atol=1e-4))
