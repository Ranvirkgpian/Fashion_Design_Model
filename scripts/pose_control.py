import os, torch, warnings, cv2
import numpy as np
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image, ImageDraw

print("=" * 55)
print("  Week 3 Task 3 — Pose Control Generation")
print("=" * 55)

# ── Step 1: Create synthetic pose maps ────────────────────
print("\n  Step 1: Creating pose maps...")

def draw_pose(filename, pose_type="standing"):
    img = Image.new("RGB", (768, 768), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    if pose_type == "standing":
        # Head
        draw.ellipse([354, 60, 414, 120], outline="white", width=3)
        # Neck
        draw.line([384, 120, 384, 155], fill="white", width=3)
        # Shoulders
        draw.line([284, 170, 484, 170], fill="white", width=3)
        # Left arm down
        draw.line([284, 170, 264, 280], fill="white", width=3)
        draw.line([264, 280, 274, 380], fill="white", width=3)
        # Right arm (hand on hip)
        draw.line([484, 170, 524, 260], fill="white", width=3)
        draw.line([524, 260, 514, 340], fill="white", width=3)
        # Spine
        draw.line([384, 155, 384, 380], fill="white", width=3)
        # Hips
        draw.line([314, 380, 454, 380], fill="white", width=3)
        # Left leg
        draw.line([314, 380, 304, 550], fill="white", width=3)
        draw.line([304, 550, 294, 700], fill="white", width=3)
        # Right leg
        draw.line([454, 380, 464, 550], fill="white", width=3)
        draw.line([464, 550, 474, 700], fill="white", width=3)
        # Joints
        for pt in [(384,120),(284,170),(484,170),(264,280),(524,260),
                   (274,380),(514,340),(314,380),(454,380),
                   (304,550),(464,550),(294,700),(474,700)]:
            draw.ellipse([pt[0]-6, pt[1]-6, pt[0]+6, pt[1]+6],
                        fill="yellow", outline="white", width=2)

    elif pose_type == "walking":
        # Head tilted slightly
        draw.ellipse([370, 55, 430, 115], outline="white", width=3)
        draw.line([400, 115, 395, 150], fill="white", width=3)
        # Shoulders slight angle
        draw.line([290, 162, 490, 178], fill="white", width=3)
        # Left arm forward
        draw.line([290, 162, 250, 280], fill="white", width=3)
        draw.line([250, 280, 230, 380], fill="white", width=3)
        # Right arm back
        draw.line([490, 178, 520, 270], fill="white", width=3)
        draw.line([520, 270, 530, 360], fill="white", width=3)
        # Spine
        draw.line([395, 150, 390, 375], fill="white", width=3)
        # Hips angled
        draw.line([310, 375, 460, 390], fill="white", width=3)
        # Left leg forward
        draw.line([310, 375, 280, 545], fill="white", width=3)
        draw.line([280, 545, 260, 700], fill="white", width=3)
        # Right leg back
        draw.line([460, 390, 490, 550], fill="white", width=3)
        draw.line([490, 550, 510, 695], fill="white", width=3)
        for pt in [(400,115),(290,162),(490,178),(250,280),(520,270),
                   (230,380),(530,360),(310,375),(460,390),
                   (280,545),(490,550),(260,700),(510,695)]:
            draw.ellipse([pt[0]-6, pt[1]-6, pt[0]+6, pt[1]+6],
                        fill="cyan", outline="white", width=2)

    img.save(rf"E:\Fashion model\outputs\{filename}")
    print(f"  ✓ Saved pose: {filename}")
    return img

pose1 = draw_pose("w3_pose_standing.png", "standing")
pose2 = draw_pose("w3_pose_walking.png",  "walking")

# ── Step 2: Load pipeline ──────────────────────────────────
print("\n  Step 2: Loading ControlNet + SDXL...")
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
print("  ✓ Pipeline ready")

negative = (
    "blurry, low quality, distorted, ugly, watermark, "
    "bad anatomy, deformed, extra limbs"
)

# ── Step 3: Generate from poses ───────────────────────────
print("\n  Step 3: Generating from pose maps...")

designs = [
    {
        "pose": pose1,
        "name": "w3_pose_result_01_standing",
        "prompt": (
            "full body fashion photograph of a young woman wearing "
            "a deep green silk evening gown, elegant standing pose, "
            "luxury fashion, white studio background, "
            "sharp focus, high resolution, vogue editorial"
        ),
    },
    {
        "pose": pose2,
        "name": "w3_pose_result_02_walking",
        "prompt": (
            "full body fashion photograph of a young woman wearing "
            "a mustard yellow maxi dress, flowing fabric, "
            "natural walking pose, white studio background, "
            "sharp focus, high resolution, editorial fashion"
        ),
    },
]

for i, d in enumerate(designs):
    print(f"\n  [{i+1}/2] Generating: {d['name']}...")
    torch.cuda.empty_cache()
    image = pipe(
        prompt=d["prompt"],
        negative_prompt=negative,
        image=d["pose"],
        controlnet_conditioning_scale=0.65,
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
print("  Task 3 Complete — Pose Control Working!")
print("  2 designs generated from skeleton pose maps")
print("=" * 55)
