import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")

import os, time, json, base64, io
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"]  = r"E:\Programe Files\huggingface"

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
from celery import Celery

# ── App setup ────────────────────────────────────────────
app = FastAPI(
    title="AI Fashion Design Assistant API",
    description="Week 7 — Async Generation Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Connections ───────────────────────────────────────────
redis_client = redis.Redis(host="localhost", port=6379, db=0)
celery_app   = Celery(broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

# ── Rate limiting (simple in-memory) ─────────────────────
RATE_LIMIT    = 10  # max requests per window
RATE_WINDOW   = 3600  # 1 hour in seconds

def check_rate_limit(ip: str) -> bool:
    key  = f"rate:{ip}"
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, RATE_WINDOW)
    return count <= RATE_LIMIT

# ── Request models ────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 25
    height: int = 768
    width: int = 768

class Sketch2DesignRequest(BaseModel):
    prompt: str
    steps: int = 25

class WardrobeRequest(BaseModel):
    category: str
    colour: str
    occasion: str
    season: str
    gender: str
    steps: int = 25

class StyleMixRequest(BaseModel):
    style_a: str
    style_b: str
    blend: float = 0.5
    gender: str = "Women"
    steps: int = 25

# ── Watermarking helper ───────────────────────────────────
def add_watermark(img_b64: str) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        text = "AI Fashion Studio"
        w, h = img.size
        draw.text(
            (w - 180, h - 30),
            text,
            fill=(255, 255, 255, 128),
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return img_b64

# ── NSFW filter (lightweight keyword check) ──────────────
NSFW_KEYWORDS = ["nude", "naked", "explicit", "nsfw", "pornographic"]

def is_nsfw_prompt(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in NSFW_KEYWORDS)

# ── Endpoints ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AI Fashion Design Assistant API", "version": "1.0.0", "week": 7}

@app.get("/health")
def health():
    try:
        redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": redis_ok,
        "timestamp": time.time(),
    }

@app.post("/generate/image")
async def generate_image(req: GenerateRequest, request: Request):
    ip = request.client.host
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 10 requests per hour.")
    if is_nsfw_prompt(req.prompt):
        raise HTTPException(status_code=400, detail="Prompt rejected by content filter.")

    task = celery_app.send_task(
        "generate_fashion_image",
        args=[req.prompt, req.steps, req.height, req.width],
    )
    return {"job_id": task.id, "status": "queued", "message": "Generation started"}

@app.post("/generate/wardrobe")
async def generate_wardrobe(req: WardrobeRequest, request: Request):
    ip = request.client.host
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    gender_str = req.gender.lower()
    prompt = (
        f"full body fashion photograph of a {gender_str} wearing "
        f"{req.colour.lower()} {req.category.lower()}, "
        f"{req.occasion.lower()} style, {req.season.lower()} season, "
        f"white studio background, professional fashion photography, "
        f"sharp focus, high resolution"
    )
    task = celery_app.send_task(
        "generate_fashion_image",
        args=[prompt, req.steps, 768, 768],
    )
    return {"job_id": task.id, "status": "queued", "prompt": prompt}

@app.post("/generate/stylemix")
async def generate_stylemix(req: StyleMixRequest, request: Request):
    ip = request.client.host
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    blend = req.blend
    gender_str = req.gender.lower()
    if blend <= 0.3:
        mixed = req.style_a
    elif blend >= 0.7:
        mixed = req.style_b
    else:
        mixed = f"blend of {req.style_a} and {req.style_b}"

    prompt = (
        f"full body fashion photograph of a {gender_str} wearing "
        f"{mixed}, white studio background, "
        f"professional fashion photography, sharp focus, high resolution"
    )
    task = celery_app.send_task(
        "generate_fashion_image",
        args=[prompt, req.steps, 768, 768],
    )
    return {"job_id": task.id, "status": "queued", "prompt": prompt}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    result = celery_app.AsyncResult(job_id)
    state  = result.state

    if state == "PENDING":
        return {"job_id": job_id, "status": "pending", "progress": 0}
    elif state == "PROGRESS":
        meta = result.info or {}
        return {
            "job_id":   job_id,
            "status":   "processing",
            "progress": meta.get("progress", 0),
            "message":  meta.get("message", ""),
        }
    elif state == "SUCCESS":
        res = result.get()
        img_b64 = res.get("image_base64", "")
        if img_b64:
            img_b64 = add_watermark(img_b64)
        return {
            "job_id":       job_id,
            "status":       "completed",
            "progress":     100,
            "image_base64": img_b64,
            "prompt":       res.get("prompt", ""),
        }
    elif state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(result.info)}
    else:
        return {"job_id": job_id, "status": state.lower(), "progress": 0}

@app.get("/jobs/stats")
def job_stats():
    inspect = celery_app.control.inspect()
    active  = inspect.active() or {}
    reserved = inspect.reserved() or {}
    total_active   = sum(len(v) for v in active.values())
    total_reserved = sum(len(v) for v in reserved.values())
    return {
        "active_jobs":  total_active,
        "queued_jobs":  total_reserved,
        "redis_keys":   redis_client.dbsize(),
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
