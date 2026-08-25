import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, torch
from transformers import CLIPProcessor, CLIPModel

os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
model.eval()

inputs = processor(text=["a black tshirt"], return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    out = model.get_text_features(**inputs)

print("Attributes:", [a for a in dir(out) if not a.startswith("_")])
print()
print("pooler_output is tensor:", torch.is_tensor(out.pooler_output) if hasattr(out, "pooler_output") else "N/A")
if hasattr(out, "pooler_output"):
    print("pooler_output shape:", out.pooler_output.shape)
print()
print("last_hidden_state shape:", out.last_hidden_state.shape if hasattr(out, "last_hidden_state") else "N/A")

# Also check the full model forward pass approach (this is what worked for images)
print()
print("--- Testing full model forward approach ---")
text_inputs = processor(text=["a black tshirt"], images=None, return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    text_only_out = model.text_model(**{k:v for k,v in text_inputs.items() if k in ["input_ids","attention_mask"]})
    pooled = text_only_out.pooler_output
    projected = model.text_projection(pooled)
    print("Projected shape:", projected.shape)
    print("First 5 values:", projected[0][:5])
