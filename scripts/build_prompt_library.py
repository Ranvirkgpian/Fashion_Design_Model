import json, os

TEMPLATE_FILE = r"E:\Fashion model\notebooks\prompt_templates.json"
os.makedirs(os.path.dirname(TEMPLATE_FILE), exist_ok=True)

templates = {
    "base_structure": (
        "full body fashion photograph of a {gender} wearing {description}, "
        "{style} style, {background} background, "
        "professional fashion photography, sharp focus, high resolution"
    ),
    "quality_boosters": [
        "vogue editorial style", "sharp focus", "high resolution",
        "professional fashion photography", "soft studio lighting",
        "8k uhd", "detailed fabric texture"
    ],
    "negative_prompt": (
        "blurry, low quality, distorted, ugly, watermark, "
        "bad anatomy, deformed, extra limbs, disfigured, "
        "out of frame, fabric pattern only, no model"
    ),
    "by_category": {
        "Topwear": {
            "women": "full body fashion photograph of a young woman wearing {color} {article_type}, {usage} style, white studio background, professional fashion photography, sharp focus, high resolution, editorial",
            "men":   "full body fashion photograph of a young man wearing {color} {article_type}, {usage} style, white studio background, professional fashion photography, sharp focus, high resolution"
        },
        "Bottomwear": {
            "women": "full body fashion photograph of a young woman wearing {color} {article_type} with a white top, {usage} style, white studio background, professional fashion photography, sharp focus",
            "men":   "full body fashion photograph of a young man wearing {color} {article_type} with a white shirt, {usage} style, white studio background, professional fashion photography, sharp focus"
        },
        "Ethnic": {
            "women": "full body fashion photograph of a young indian woman wearing {color} {article_type}, traditional ethnic wear, {usage} style, white studio background, professional photography, sharp focus",
            "men":   "full body fashion photograph of a young indian man wearing {color} {article_type}, traditional ethnic wear, {usage} style, white studio background, professional photography, sharp focus"
        },
        "Footwear": {
            "all":   "close up product photograph of {color} {article_type}, white studio background, professional product photography, sharp focus, high resolution"
        },
        "Accessories": {
            "all":   "close up product photograph of {color} {article_type}, white studio background, professional product photography, sharp focus, high resolution, detailed"
        },
        "Formal": {
            "women": "full body fashion photograph of a young woman wearing {color} formal {article_type}, professional business attire, white studio background, editorial photography, sharp focus",
            "men":   "full body fashion photograph of a young man wearing {color} formal {article_type}, professional business attire, white studio background, editorial photography, sharp focus"
        }
    },
    "example_prompts": [
        {
            "category": "Topwear", "gender": "Women",
            "colour": "Red", "article": "Tshirts", "usage": "Casual",
            "prompt": "full body fashion photograph of a young woman wearing red casual tshirt, high waist jeans, casual style, white studio background, professional fashion photography, sharp focus, high resolution"
        },
        {
            "category": "Ethnic", "gender": "Men",
            "colour": "Navy Blue", "article": "Kurtas", "usage": "Casual",
            "prompt": "full body fashion photograph of a young indian man wearing navy blue cotton kurta with white pyjama, traditional ethnic wear, casual style, white studio background, professional fashion photography, sharp focus, high resolution"
        },
        {
            "category": "Formal", "gender": "Men",
            "colour": "Charcoal Grey", "article": "Suits", "usage": "Formal",
            "prompt": "full body fashion photograph of a young man wearing charcoal grey slim fit formal suit, crisp white shirt, confident pose, white studio background, editorial fashion photography, sharp focus, high resolution"
        },
        {
            "category": "Bottomwear", "gender": "Women",
            "colour": "Black", "article": "Jeans", "usage": "Casual",
            "prompt": "full body fashion photograph of a young woman wearing black high waist jeans with white oversized tshirt, casual streetwear style, white studio background, professional fashion photography, sharp focus"
        }
    ]
}

with open(TEMPLATE_FILE, "w") as f:
    json.dump(templates, f, indent=2)

print("=" * 55)
print("  Prompt Template Library — Saved!")
print("=" * 55)
print(f"\n  File: {TEMPLATE_FILE}")
print(f"\n  Contains:")
print(f"  ✓  Base prompt structure template")
print(f"  ✓  {len(templates['quality_boosters'])} quality booster keywords")
print(f"  ✓  {len(templates['by_category'])} category-specific templates")
print(f"  ✓  {len(templates['example_prompts'])} ready-to-use example prompts")
print(f"\n  These templates connect directly to your")
print(f"  Week 1 dataset — 143 article types x 47 colours!")
print("=" * 55)
