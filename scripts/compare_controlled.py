import os, torch, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image
import cv2

print("=" * 55)
print("  Week 3 Task 4 — Controlled vs Uncontrolled")
print("=" * 55)

PROMPT = (
    "full body fashion photograph of a young woman wearing "
    "a deep emerald green silk evening gown, flowing fabric, "
    "elegant pose, white studio background, "
    "sharp focus, high resolution, vogue editorial"
)
NEGATIVE = (
    "blurry, low quality, distorted, ugly, watermark, "
    "bad anatomy, deformed, extra limbs"
)

# ── Uncontrolled: plain SDXL ───────────────────────────────
print("\n  [1/2] Generating UNCONTROLLED image...")
pipe_plain = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe_plain.enable_model_cpu_offload()
torch.cuda.empty_cache()

img_uncontrolled = pipe_plain(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    height=768, width=768,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator().manual_seed(42),
).images[0]
img_uncontrolled.save(r"E:\Fashion model\outputs\w3_compare_uncontrolled.png")
print("  ✓ Saved uncontrolled image")

del pipe_plain
torch.cuda.empty_cache()

# ── Controlled: SDXL + ControlNet ─────────────────────────
print("\n  [2/2] Generating CONTROLLED image (same seed)...")

src = Image.open(r"E:\Fashion model\outputs\01_pink_floral_dress.png").convert("RGB").resize((768,768))
arr = np.array(src)
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 100, 200)
control_image = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

controlnet = ControlNetModel.from_pretrained(
    r"E:\Fashion model\models\controlnet-canny-sdxl",
    torch_dtype=torch.float16,
)
pipe_ctrl = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe_ctrl.enable_model_cpu_offload()
torch.cuda.empty_cache()

img_controlled = pipe_ctrl(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    image=control_image,
    controlnet_conditioning_scale=0.7,
    height=768, width=768,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator().manual_seed(42),
).images[0]
img_controlled.save(r"E:\Fashion model\outputs\w3_compare_controlled.png")
print("  ✓ Saved controlled image")

# ── Side by side comparison ────────────────────────────────
print("\n  Creating side-by-side comparison...")
comparison = Image.new("RGB", (1600, 900), color=(245, 245, 245))

img_u = img_uncontrolled.resize((740, 768))
img_c = img_controlled.resize((740, 768))
comparison.paste(img_u, (30, 66))
comparison.paste(img_c, (830, 66))

from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(comparison)
draw.rectangle([0, 0, 1600, 60], fill=(26, 31, 75))
draw.text((30, 18),  "UNCONTROLLED — Plain SDXL", fill=(240, 208, 128))
draw.text((830, 18), "CONTROLLED — SDXL + ControlNet (canny)", fill=(240, 208, 128))
draw.text((30, 840),  "No structural guidance — pose varies each run", fill=(100,100,100))
draw.text((830, 840), "Edge map from source image — pose preserved", fill=(100,100,100))

comparison.save(r"E:\Fashion model\outputs\w3_comparison_final.png")
print("  ✓ Comparison saved → w3_comparison_final.png")

print()
print("=" * 55)
print("  Task 4 Complete — Comparison Ready!")
print("  Upload w3_comparison_final.png to see the diff")
print("=" * 55)
