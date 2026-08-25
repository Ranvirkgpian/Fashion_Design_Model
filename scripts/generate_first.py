import os, torch, warnings
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"

from diffusers import StableDiffusionXLPipeline

print("Loading SDXL from cache...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# This automatically moves model parts between CPU/GPU as needed
# Prevents OOM while keeping speed
pipe.enable_model_cpu_offload()
torch.cuda.empty_cache()

print("Generating fashion image...")

prompt = (
    "A womens floral summer dress, elegant and flowing, "
    "light pink color with delicate flower patterns, "
    "fashion photography, white background, "
    "high quality, detailed fabric texture"
)
negative = "blurry, low quality, distorted, ugly, watermark"

image = pipe(
    prompt=prompt,
    negative_prompt=negative,
    height=768,
    width=768,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

out = r"E:\Fashion model\outputs\first_generation.png"
image.save(out)
print("=" * 55)
print("SUCCESS! Image saved to:", out)
print("Open outputs folder to see your first AI fashion image!")
print("=" * 55)
