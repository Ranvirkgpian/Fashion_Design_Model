import os, torch, warnings
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"

from diffusers import StableDiffusionXLPipeline

print("Loading SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.enable_model_cpu_offload()
torch.cuda.empty_cache()

negative = (
    "blurry, low quality, distorted, ugly, watermark, "
    "bad anatomy, deformed, fabric pattern only, no model, "
    "extra limbs, disfigured, out of frame"
)

prompts = [
    {
        "name": "01_pink_floral_dress",
        "prompt": (
            "full body fashion photograph of a young woman wearing "
            "a light pink floral summer dress, flowing chiffon fabric, "
            "elegant standing pose, pure white studio background, "
            "professional fashion photography, sharp focus, "
            "high resolution, vogue editorial style, soft lighting"
        )
    },
    {
        "name": "02_navy_kurta_men",
        "prompt": (
            "full body fashion photograph of a young indian man wearing "
            "a navy blue cotton kurta with white pyjama, "
            "traditional ethnic wear, relaxed standing pose, "
            "pure white studio background, professional fashion photography, "
            "sharp focus, high resolution, clean look"
        )
    },
    {
        "name": "03_charcoal_suit",
        "prompt": (
            "full body fashion photograph of a man wearing "
            "a charcoal grey slim fit formal business suit, "
            "crisp white shirt, no tie, confident standing pose, "
            "pure white studio background, professional photography, "
            "sharp focus, high resolution, editorial menswear"
        )
    },
    {
        "name": "04_streetwear_women",
        "prompt": (
            "full body fashion photograph of a young woman wearing "
            "an oversized white graphic tshirt, high waist black jeans, "
            "white sneakers, streetwear casual style, "
            "relaxed pose, pure white studio background, "
            "professional photography, sharp focus, high resolution"
        )
    },
]

print(f"Generating {len(prompts)} fashion images...")
print()

for i, p in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] Generating: {p['name']}...")
    torch.cuda.empty_cache()
    image = pipe(
        prompt=p["prompt"],
        negative_prompt=negative,
        height=768,
        width=768,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    out = rf"E:\Fashion model\outputs\{p['name']}.png"
    image.save(out)
    print(f"       Saved → {out}")
    print()

print("=" * 55)
print("  Task 2 Complete — Prompt Engineering Results")
print("=" * 55)
print("  4 fashion images generated:")
print("  01 — Pink floral dress (women)")
print("  02 — Navy kurta (men, ethnic)")
print("  03 — Charcoal suit (men, formal)")
print("  04 — Streetwear tshirt (women, casual)")
print()
print("  Open outputs\\ folder to review all images!")
print("=" * 55)
