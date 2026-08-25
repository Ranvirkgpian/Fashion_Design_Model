import os, torch
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel

# ── ENV SETUP ──────────────────────────────────────────────────────────────
os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"
OUTPUT_DIR = r"E:\Fashion model\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Testing Custom LoRA (Fixed Weights Injection)")
print("=" * 60)

# 1. Load the Base Model
print("Loading base SDXL model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    local_files_only=True
)

# 2. Inject YOUR Custom LoRA directly into the UNet structure
print("Injecting custom lux_street LoRA weights directly into the UNet layers...")
LORA_PATH = r"E:\Fashion model\models\lora_lux_street"
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH, adapter_name="default")

# Optimize for VRAM during generation
pipe.enable_model_cpu_offload()

# 3. Generate using your Trigger Word
prompts = [
    {
        "name": "lora_test_01_womens_streetwear_fixed",
        "prompt": "lux_street style, full body fashion photograph of a young woman wearing a black oversized tshirt and wide leg charcoal trousers, minimalist luxury streetwear, clean lines, white studio background, editorial fashion, high resolution"
    },
    {
        "name": "lora_test_02_mens_streetwear_fixed",
        "prompt": "lux_street style, full body fashion photograph of a young man wearing a crisp white unstructured blazer and beige relaxed trousers, minimalist luxury streetwear, high end fashion, white studio background, editorial, sharp focus"
    }
    ]

negative_prompt = "blurry, low quality, distorted, ugly, watermark, bad anatomy, over-accessorized, messy, vibrant colors"

print("\nGenerating new designs with your trained style applied...")
for p in prompts:
    print(f"Generating {p['name']}...")
    image = pipe(
        prompt=p["prompt"],
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    
    save_path = os.path.join(OUTPUT_DIR, f"{p['name']}.png")
    image.save(save_path)
    print(f"  ✓ Saved to {save_path}")

print("\n" + "=" * 60)
print("  Inference Complete! Your real custom style images are ready.")
print("=" * 60)
