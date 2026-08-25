import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")

import os, torch, warnings, json
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"]  = "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"
os.environ["HF_HOME"]   = r"E:\Programe Files\huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Lazy-load models (load once on first use) ──────────────
_sdxl_pipe = None
_controlnet_pipe = None
_clip_model = None
_clip_processor = None
_chroma_collection = None

def get_sdxl():
    global _sdxl_pipe
    if _sdxl_pipe is None:
        from diffusers import StableDiffusionXLPipeline
        print("Loading SDXL...")
        _sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            local_files_only=True,
        )
        _sdxl_pipe.enable_model_cpu_offload()
        print("SDXL ready")
    return _sdxl_pipe

def get_controlnet():
    global _controlnet_pipe
    if _controlnet_pipe is None:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        print("Loading ControlNet...")
        cn = ControlNetModel.from_pretrained(
            r"E:\Fashion model\models\controlnet-canny-sdxl",
            torch_dtype=torch.float16,
        )
        _controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=cn,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            local_files_only=True,
        )
        _controlnet_pipe.enable_model_cpu_offload()
        print("ControlNet ready")
    return _controlnet_pipe

def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        _clip_model.eval()
    return _clip_model, _clip_processor

def get_chroma():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=r"E:\Fashion model\data\chromadb")
        _chroma_collection = client.get_collection("fashion_styles")
    return _chroma_collection

NEGATIVE = "blurry, low quality, distorted, ugly, watermark, bad anatomy, deformed"

# ── Tab 1: Sketch2Design ──────────────────────────────────
def sketch2design(input_image, prompt, strength, steps):
    if input_image is None:
        return None, "Please upload an image or sketch first"
    try:
        torch.cuda.empty_cache()
        img = Image.fromarray(input_image).convert("RGB").resize((768, 768))
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        control = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        if not prompt.strip():
            prompt = "full body fashion photograph of a woman wearing an elegant dress, white studio background, high quality"

        pipe = get_controlnet()
        result = pipe(
            prompt=prompt + ", fashion photography, high resolution",
            negative_prompt=NEGATIVE,
            image=control,
            controlnet_conditioning_scale=float(strength),
            height=768, width=768,
            num_inference_steps=int(steps),
            guidance_scale=7.5,
        ).images[0]
        torch.cuda.empty_cache()
        return result, "Generated successfully!"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ── Tab 2: StyleMixer ─────────────────────────────────────
def stylemixer(style_a, style_b, blend, gender, steps):
    try:
        torch.cuda.empty_cache()
        blend = float(blend)
        gender_str = gender.lower() if gender else "women"

        if blend <= 0.3:
            mixed = f"{style_a}, {gender_str} fashion"
        elif blend >= 0.7:
            mixed = f"{style_b}, {gender_str} fashion"
        else:
            mixed = f"blend of {style_a} and {style_b}, {gender_str} fashion"

        prompt = (
            f"full body fashion photograph of a {gender_str} wearing "
            f"{mixed}, white studio background, "
            f"professional fashion photography, sharp focus, high resolution"
        )

        pipe = get_sdxl()
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            height=768, width=768,
            num_inference_steps=int(steps),
            guidance_scale=7.5,
        ).images[0]
        torch.cuda.empty_cache()
        return result, f"Prompt used: {prompt}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ── Tab 3: WardrobeGen ────────────────────────────────────
def wardrobegen(category, colour, occasion, season, gender, steps):
    try:
        torch.cuda.empty_cache()
        gender_str = gender.lower()

        if category in ["Tshirts","Shirts","Tops","Jackets","Sweatshirts"]:
            garment_desc = f"{colour.lower()} {category.lower()}"
        elif category in ["Jeans","Trousers","Shorts","Skirts","Leggings"]:
            garment_desc = f"{colour.lower()} {category.lower()}"
        else:
            garment_desc = f"{colour.lower()} {category.lower()}"

        prompt = (
            f"full body fashion photograph of a {gender_str} wearing "
            f"{garment_desc}, {occasion.lower()} style, "
            f"appropriate for {season.lower()} season, "
            f"white studio background, professional fashion photography, "
            f"sharp focus, high resolution, editorial"
        )

        pipe = get_sdxl()
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            height=768, width=768,
            num_inference_steps=int(steps),
            guidance_scale=7.5,
        ).images[0]
        torch.cuda.empty_cache()
        return result, f"Generated: {garment_desc} for {gender_str}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ── Tab 4: Trend Forecaster ───────────────────────────────
def trend_forecast(query_text, gender_filter, n_results):
    try:
        model, processor = get_clip()
        inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_out = model.text_model(input_ids=inputs["input_ids"],
                                         attention_mask=inputs["attention_mask"])
            projected = model.text_projection(text_out.pooler_output)
            projected = projected / projected.norm(dim=-1, keepdim=True)
        query_emb = projected.cpu().numpy().tolist()

        collection = get_chroma()
        where = {"gender": gender_filter} if gender_filter != "All" else None
        results = collection.query(query_embeddings=query_emb, n_results=int(n_results), where=where)

        output = f"Top {n_results} recommendations for: \"{query_text}\"\n"
        output += "=" * 60 + "\n"
        for i, (doc, dist, meta) in enumerate(zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        )):
            sim = 1 - dist
            output += f"\n{i+1}. [{sim:.3f}] {doc}\n"
            output += f"   Trend: {meta.get('trend_category','N/A')}  |  Score: {meta.get('trend_score','N/A')}\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

# ── Build Gradio UI ───────────────────────────────────────
print("Building Gradio Creative Studio...")

with gr.Blocks(
    title="AI Fashion Creative Studio",
    theme=gr.themes.Base(
        primary_hue="purple",
        secondary_hue="teal",
    )
) as demo:

    gr.Markdown("""
    # AI Fashion Creative Studio
    ### Week 6 — Generative AI Fashion Assistant · H&P Projects
    """)

    with gr.Tabs():

        # ── Tab 1: Sketch2Design ──────────────────────────
        with gr.Tab("Sketch2Design"):
            gr.Markdown("### Upload a sketch or fashion image → AI generates a styled design")
            with gr.Row():
                with gr.Column(scale=1):
                    sketch_input = gr.Image(label="Upload sketch or image", type="numpy")
                    sketch_prompt = gr.Textbox(
                        label="Style prompt",
                        placeholder="full body fashion photograph of a woman wearing an elegant red gown...",
                        lines=3
                    )
                    sketch_strength = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="ControlNet strength")
                    sketch_steps   = gr.Slider(10, 40, value=25, step=5, label="Inference steps")
                    sketch_btn     = gr.Button("Generate Design", variant="primary")
                with gr.Column(scale=1):
                    sketch_output = gr.Image(label="Generated Design")
                    sketch_status = gr.Textbox(label="Status", interactive=False)
            sketch_btn.click(sketch2design, inputs=[sketch_input, sketch_prompt, sketch_strength, sketch_steps],
                             outputs=[sketch_output, sketch_status])

        # ── Tab 2: StyleMixer ─────────────────────────────
        with gr.Tab("StyleMixer"):
            gr.Markdown("### Blend two fashion styles with a mix slider")
            with gr.Row():
                with gr.Column(scale=1):
                    style_a = gr.Textbox(label="Style A", value="minimalist luxury streetwear, neutral tones, clean silhouette")
                    style_b = gr.Textbox(label="Style B", value="bohemian ethnic prints, colorful embroidery, flowing fabric")
                    blend   = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Blend (0 = Style A, 1 = Style B)")
                    mix_gender = gr.Radio(["Women","Men"], label="Gender", value="Women")
                    mix_steps  = gr.Slider(10, 40, value=25, step=5, label="Inference steps")
                    mix_btn    = gr.Button("Mix Styles", variant="primary")
                with gr.Column(scale=1):
                    mix_output = gr.Image(label="Mixed Style Output")
                    mix_prompt_out = gr.Textbox(label="Generated prompt", interactive=False)
            mix_btn.click(stylemixer, inputs=[style_a, style_b, blend, mix_gender, mix_steps],
                          outputs=[mix_output, mix_prompt_out])

        # ── Tab 3: WardrobeGen ────────────────────────────
        with gr.Tab("WardrobeGen"):
            gr.Markdown("### Generate any garment from your Week 1 vocabulary")
            with gr.Row():
                with gr.Column(scale=1):
                    wg_category = gr.Dropdown(
                        ["Tshirts","Shirts","Jeans","Trousers","Jackets","Dresses",
                         "Tops","Skirts","Sweatshirts","Shorts","Kurtas","Leggings"],
                        label="Article Type", value="Tshirts"
                    )
                    wg_colour = gr.Dropdown(
                        ["Black","White","Navy Blue","Grey","Beige","Red",
                         "Green","Pink","Charcoal","Brown","Off White"],
                        label="Base Colour", value="Black"
                    )
                    wg_occasion = gr.Dropdown(
                        ["Casual","Formal","Sports","Ethnic","Party","Smart Casual"],
                        label="Occasion", value="Casual"
                    )
                    wg_season = gr.Dropdown(
                        ["Summer","Winter","Fall","Spring"],
                        label="Season", value="Summer"
                    )
                    wg_gender = gr.Radio(["Women","Men","Unisex"], label="Gender", value="Women")
                    wg_steps  = gr.Slider(10, 40, value=25, step=5, label="Inference steps")
                    wg_btn    = gr.Button("Generate Outfit", variant="primary")
                with gr.Column(scale=1):
                    wg_output = gr.Image(label="Generated Outfit")
                    wg_status = gr.Textbox(label="Status", interactive=False)
            wg_btn.click(wardrobegen, inputs=[wg_category, wg_colour, wg_occasion, wg_season, wg_gender, wg_steps],
                         outputs=[wg_output, wg_status])

        # ── Tab 4: Trend Forecaster ───────────────────────
        with gr.Tab("Trend Forecaster"):
            gr.Markdown("### Query the ChromaDB fashion knowledge base for trend recommendations")
            with gr.Row():
                with gr.Column(scale=1):
                    tf_query  = gr.Textbox(
                        label="Style query",
                        placeholder="cozy winter layering pieces in neutral tones",
                        lines=2
                    )
                    tf_gender = gr.Radio(["All","Men","Women","Unisex"], label="Gender filter", value="All")
                    tf_n      = gr.Slider(3, 10, value=5, step=1, label="Number of results")
                    tf_btn    = gr.Button("Find Trends", variant="primary")
                with gr.Column(scale=1):
                    tf_output = gr.Textbox(label="Recommendations", lines=18, interactive=False)
            tf_btn.click(trend_forecast, inputs=[tf_query, tf_gender, tf_n], outputs=[tf_output])

print("Launching Creative Studio on http://localhost:7860")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
