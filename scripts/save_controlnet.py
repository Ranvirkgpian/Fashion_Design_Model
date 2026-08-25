import os, torch, warnings
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"

from diffusers import ControlNetModel

print("Loading ControlNet from E: cache...")
model = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
save_path = r"E:\Fashion model\models\controlnet-canny-sdxl"
model.save_pretrained(save_path)
print("=" * 50)
print("Saved to:", save_path)
print("=" * 50)
