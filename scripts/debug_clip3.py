import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
model.eval()

img_dir = r"E:\Fashion model\data\deepfashion\raw\images"
img_path = os.path.join(img_dir, os.listdir(img_dir)[0])
image = Image.open(img_path).convert("RGB")

img_inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    raw_out = model.get_image_features(**img_inputs)

print("Type:", type(raw_out))
print("Is tensor:", torch.is_tensor(raw_out))
if hasattr(raw_out, "pooler_output"):
    print("Has pooler_output, shape:", raw_out.pooler_output.shape)

# Correct manual projection
with torch.no_grad():
    vision_out = model.vision_model(pixel_values=img_inputs["pixel_values"])
    projected = model.visual_projection(vision_out.pooler_output)
    projected = projected / projected.norm(dim=-1, keepdim=True)
print("\nCorrect projected embedding shape:", projected.shape)
print("First 5 values:", projected[0][:5].cpu().numpy())
