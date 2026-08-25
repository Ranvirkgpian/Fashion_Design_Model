import os, torch, warnings, cv2
import numpy as np
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image

print("=" * 55)
print("  Week 3 Task 2 — Sketch to Design Generation")
print("=" * 55)

# ── Step 1: Load source image and extract edges ────────────
print("\n  Step 1: Extracting edge map from dress image...")
src = Image.open(r"E:\Fashion model\outputs\01_pink_floral_dress.png").convert("RGB")
src = src.resize((768, 768))

img_array = np.array(src)
gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, threshold1=100, threshold2=200)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
control_image = Image.fromarray(edges_rgb)
control_image.save(r"E:\Fashion model\outputs\w3_edge_map.png")
print("  ✓ Edge map saved → outputs\w3_edge_map.png")

# ── Step 2: Load ControlNet + SDXL ────────────────────────
print("\n  Step 2: Loading ControlNet + SDXL pipeline...")
controlnet = ControlNetModel.from_pretrained(
    r"E:\Fashion model\models\controlnet-canny-sdxl",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.enable_model_cpu_offload()
torch.cuda.empty_cache()
print("  ✓ Pipeline ready on RTX 4060")

# ── Step 3: Generate 3 styles from same edge map ──────────
print("\n  Step 3: Generating 3 fashion styles from same structure...")

designs = [
    {
        "name": "w3_design_01_red_gown",
        "prompt": (
            "full body fashion photograph of a young woman wearing "
            "an elegant red evening gown, flowing silk fabric, "
            "luxury fashion, white studio background, "
            "sharp focus, high resolution, vogue editorial"
        ),
    },
    {
        "name": "w3_design_02_blue_dress",
        "prompt": (
            "full body fashion photograph of a young woman wearing "
            "a royal blue cocktail dress, structured fabric, "
            "modern fashion, white studio background, "
            "sharp focus, high resolution, editorial style"
        ),
    },
    {
        "name": "w3_design_03_black_dress",
        "prompt": (
            "full body fashion photograph of a young woman wearing "
            "a sleek black minimal dress, matte fabric, "
            "high fashion editorial, white studio background, "
            "sharp focus, high resolution, luxury brand"
        ),
    },
]

negative = (
    "blurry, low quality, distorted, ugly, watermark, "
    "bad anatomy, deformed, extra limbs"
)

for i, d in enumerate(designs):
    print(f"\n  [{i+1}/3] Generating: {d['name']}...")
    torch.cuda.empty_cache()
    image = pipe(
        prompt=d["prompt"],
        negative_prompt=negative,
        image=control_image,
        controlnet_conditioning_scale=0.7,
        height=768,
        width=768,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    out = rf"E:\Fashion model\outputs\{d['name']}.png"
    image.save(out)
    print(f"  ✓ Saved → {out}")

print()
print("=" * 55)
print("  Week 3 Task 2 Complete!")
print("  Generated 3 designs from the same edge structure")
print("  Open outputs\\ to compare all results")
print("=" * 55)
