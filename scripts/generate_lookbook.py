import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")

import os, torch, warnings, json
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"]  = r"E:\Programe Files\huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from diffusers import StableDiffusionXLPipeline
from PIL import Image

print("=" * 60)
print("  Week 8 Task 3 — AI Fashion Lookbook Portfolio")
print("  Generating 24 curated designs")
print("=" * 60)

OUT_DIR = r"E:\Fashion model\outputs\lookbook"
os.makedirs(OUT_DIR, exist_ok=True)

print("\n  Loading SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    local_files_only=True,
)
pipe.enable_model_cpu_offload()
print("  SDXL ready\n")

NEGATIVE = "blurry, low quality, distorted, ugly, watermark, bad anatomy, deformed"

# ── 24 curated designs across 6 collections ────────────────
lookbook = [
    # Collection 1: Evening Elegance
    {"collection": "Evening Elegance", "name": "01_emerald_gown",
     "prompt": "full body fashion photograph of a woman wearing an emerald green silk evening gown, elegant draping, standing pose, white studio background, professional fashion photography, sharp focus, high resolution, vogue editorial"},
    {"collection": "Evening Elegance", "name": "02_burgundy_dress",
     "prompt": "full body fashion photograph of a woman wearing a burgundy velvet evening dress, off-shoulder, elegant pose, white studio background, professional fashion photography, sharp focus, high resolution"},
    {"collection": "Evening Elegance", "name": "03_champagne_gown",
     "prompt": "full body fashion photograph of a woman wearing a champagne satin gown, flowing train, elegant pose, white studio background, professional fashion photography, sharp focus, high resolution"},
    {"collection": "Evening Elegance", "name": "04_navy_cocktail",
     "prompt": "full body fashion photograph of a woman wearing a navy blue cocktail dress, structured silhouette, white studio background, professional fashion photography, sharp focus, high resolution"},

    # Collection 2: Minimalist Streetwear
    {"collection": "Minimalist Streetwear", "name": "05_lux_street_black",
     "prompt": "full body fashion photograph of a woman wearing lux_street style, black oversized tshirt, wide leg trousers, minimalist luxury streetwear, white studio background, professional fashion photography, sharp focus, high resolution"},
    {"collection": "Minimalist Streetwear", "name": "06_lux_street_beige",
     "prompt": "full body fashion photograph of a man wearing lux_street style, beige oversized hoodie, tapered trousers, minimalist luxury streetwear, white studio background, professional fashion photography, sharp focus, high resolution"},
    {"collection": "Minimalist Streetwear", "name": "07_lux_street_grey",
     "prompt": "full body fashion photograph of a woman wearing lux_street style, grey cropped jacket, straight leg pants, clean silhouette, white studio background, professional fashion photography, sharp focus"},
    {"collection": "Minimalist Streetwear", "name": "08_lux_street_white",
     "prompt": "full body fashion photograph of a man wearing lux_street style, white minimal tshirt, black tailored joggers, neutral tones, white studio background, professional fashion photography, sharp focus"},

    # Collection 3: Ethnic Heritage
    {"collection": "Ethnic Heritage", "name": "09_navy_kurta",
     "prompt": "full body fashion photograph of a man wearing navy blue cotton kurta with white pyjama, traditional indian ethnic wear, white studio background, professional fashion photography, sharp focus, high resolution"},
    {"collection": "Ethnic Heritage", "name": "10_maroon_saree",
     "prompt": "full body fashion photograph of a woman wearing a maroon silk saree with gold embroidery, traditional indian wear, elegant pose, white studio background, professional fashion photography, sharp focus"},
    {"collection": "Ethnic Heritage", "name": "11_green_sherwani",
     "prompt": "full body fashion photograph of a man wearing an emerald green sherwani with gold buttons, traditional formal indian wear, white studio background, professional fashion photography, sharp focus"},
    {"collection": "Ethnic Heritage", "name": "12_pink_lehenga",
     "prompt": "full body fashion photograph of a woman wearing a pink embroidered lehenga, traditional indian bridal wear, elegant pose, white studio background, professional fashion photography, sharp focus"},

    # Collection 4: Business Formal
    {"collection": "Business Formal", "name": "13_charcoal_suit",
     "prompt": "full body fashion photograph of a man wearing a charcoal grey slim fit business suit, white shirt, confident pose, white studio background, editorial fashion photography, sharp focus, high resolution"},
    {"collection": "Business Formal", "name": "14_navy_blazer",
     "prompt": "full body fashion photograph of a woman wearing a navy blue tailored blazer with matching trousers, professional business attire, white studio background, editorial photography, sharp focus"},
    {"collection": "Business Formal", "name": "15_black_pantsuit",
     "prompt": "full body fashion photograph of a woman wearing a black formal pantsuit, structured shoulders, professional business attire, white studio background, editorial photography, sharp focus"},
    {"collection": "Business Formal", "name": "16_grey_three_piece",
     "prompt": "full body fashion photograph of a man wearing a grey three-piece formal suit, vest and tie, white studio background, editorial fashion photography, sharp focus, high resolution"},

    # Collection 5: Casual Everyday
    {"collection": "Casual Everyday", "name": "17_denim_jacket",
     "prompt": "full body fashion photograph of a woman wearing a blue denim jacket with white tshirt and black jeans, casual streetwear, white studio background, professional fashion photography, sharp focus"},
    {"collection": "Casual Everyday", "name": "18_khaki_chinos",
     "prompt": "full body fashion photograph of a man wearing khaki chinos with a light blue casual shirt, relaxed pose, white studio background, professional fashion photography, sharp focus"},
    {"collection": "Casual Everyday", "name": "19_floral_sundress",
     "prompt": "full body fashion photograph of a woman wearing a floral print sundress, casual summer style, white studio background, professional fashion photography, sharp focus, high resolution"},
    {"collection": "Casual Everyday", "name": "20_striped_polo",
     "prompt": "full body fashion photograph of a man wearing a navy striped polo shirt with beige shorts, casual summer style, white studio background, professional fashion photography, sharp focus"},

    # Collection 6: Sport & Athleisure
    {"collection": "Sport & Athleisure", "name": "21_black_tracksuit",
     "prompt": "full body fashion photograph of a woman wearing a black athletic tracksuit, sporty style, white studio background, professional fashion photography, sharp focus, high resolution"},
    {"collection": "Sport & Athleisure", "name": "22_grey_joggers",
     "prompt": "full body fashion photograph of a man wearing grey joggers with a fitted athletic top, sporty casual style, white studio background, professional fashion photography, sharp focus"},
    {"collection": "Sport & Athleisure", "name": "23_yoga_set",
     "prompt": "full body fashion photograph of a woman wearing a matching lavender yoga set, athletic wear, white studio background, professional fashion photography, sharp focus, high resolution"},
    {"collection": "Sport & Athleisure", "name": "24_running_gear",
     "prompt": "full body fashion photograph of a man wearing black and red running gear, athletic performance wear, white studio background, professional fashion photography, sharp focus"},
]

results_log = []
for i, item in enumerate(lookbook):
    print(f"  [{i+1}/24] {item['collection']}: {item['name']}...")
    torch.cuda.empty_cache()
    try:
        image = pipe(
            prompt=item["prompt"],
            negative_prompt=NEGATIVE,
            height=768, width=768,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        out_path = os.path.join(OUT_DIR, f"{item['name']}.png")
        image.save(out_path)
        results_log.append({"file": f"{item['name']}.png", "collection": item["collection"], "status": "success"})
        print(f"        Saved: {item['name']}.png")
    except Exception as e:
        results_log.append({"file": item["name"], "collection": item["collection"], "status": f"failed: {e}"})
        print(f"        Failed: {e}")

with open(os.path.join(OUT_DIR, "lookbook_manifest.json"), "w") as f:
    json.dump(results_log, f, indent=2)

success_count = sum(1 for r in results_log if r["status"] == "success")
print()
print("=" * 60)
print(f"  Lookbook generation complete!")
print(f"  {success_count}/24 designs generated successfully")
print(f"  Saved to: {OUT_DIR}")
print("=" * 60)
