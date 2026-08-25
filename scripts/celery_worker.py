import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")

import os
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"]  = r"E:\Programe Files\huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from celery import Celery
import torch, warnings, base64, io, time
warnings.filterwarnings("ignore")

app = Celery(
    "fashion_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEGATIVE = "blurry, low quality, distorted, ugly, watermark, bad anatomy, deformed"

_sdxl_pipe = None

def get_sdxl():
    global _sdxl_pipe
    if _sdxl_pipe is None:
        from diffusers import StableDiffusionXLPipeline
        print("[Worker] Loading SDXL...")
        _sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            local_files_only=True,
        )
        _sdxl_pipe.enable_model_cpu_offload()
        print("[Worker] SDXL ready")
    return _sdxl_pipe

@app.task(bind=True, name="generate_fashion_image")
def generate_fashion_image(self, prompt, steps=25, height=768, width=768):
    """
    Async task: generate a fashion image from a text prompt.
    Reports progress back to Redis via self.update_state.
    """
    try:
        self.update_state(state="PROGRESS", meta={"progress": 5, "message": "Loading model..."})
        pipe = get_sdxl()
        torch.cuda.empty_cache()

        self.update_state(state="PROGRESS", meta={"progress": 15, "message": "Starting generation..."})

        def progress_callback(pipe_obj, step, timestep, callback_kwargs):
            pct = 15 + int((step / steps) * 70)
            self.update_state(state="PROGRESS", meta={"progress": pct, "message": f"Step {step}/{steps}"})
            return callback_kwargs

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            height=height, width=width,
            num_inference_steps=steps,
            guidance_scale=7.5,
            callback_on_step_end=progress_callback,
        )
        image = result.images[0]

        self.update_state(state="PROGRESS", meta={"progress": 90, "message": "Encoding result..."})

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        torch.cuda.empty_cache()
        return {
            "status": "completed",
            "progress": 100,
            "image_base64": img_b64,
            "prompt": prompt,
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    print("Starting Celery worker...")
    app.worker_main(["worker", "--loglevel=info", "--pool=solo"])
