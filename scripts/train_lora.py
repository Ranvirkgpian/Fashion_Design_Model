import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")

import os, torch, warnings, json, random, gc
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"]  = r"E:\Programe Files\huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

DATASET_DIR = r"E:\Fashion model\data\lora_dataset\minimalist_streetwear"
OUTPUT_DIR  = r"E:\Fashion model\models\lora_lux_street"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_STEPS   = 300
LEARNING_RATE = 1e-4
GRAD_ACCUM    = 4
LORA_RANK     = 2
LORA_ALPHA    = 4
RESOLUTION    = 512
DEVICE        = "cuda"

torch.cuda.empty_cache()
print("=" * 55)
print("  Week 4 Task 2 — LoRA Fine-Tuning")
print("=" * 55)

# ── Dataset ───────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
imgs, caps = [], []
for f in sorted(os.listdir(DATASET_DIR)):
    if not f.endswith(".jpg"): continue
    txt = os.path.join(DATASET_DIR, f.replace(".jpg",".txt"))
    if not os.path.exists(txt): continue
    imgs.append(os.path.join(DATASET_DIR, f))
    with open(txt) as fp: caps.append(fp.read().strip())
print(f"  {len(imgs)} pairs")

tf = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

# ── Load SDXL fp16 from cache ─────────────────────────────
print("\n[2/5] Loading SDXL fp16 from cache...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    local_files_only=True,
)
tok1      = pipe.tokenizer
tok2      = pipe.tokenizer_2
text_enc1 = pipe.text_encoder
text_enc2 = pipe.text_encoder_2
vae       = pipe.vae.to(DEVICE)
unet      = pipe.unet
noise_sch = DDPMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="scheduler", local_files_only=True,
)
text_enc1.requires_grad_(False)
text_enc2.requires_grad_(False)
vae.requires_grad_(False)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Encode everything before training ─────────────────────
print("\n[3/5] Encoding captions + images...")
caption_cache = {}
for cap in set(caps):
    t1 = tok1([cap], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    t2 = tok2([cap], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.no_grad():
        e1 = text_enc1(t1.input_ids, output_hidden_states=True)
        e2 = text_enc2(t2.input_ids, output_hidden_states=True)
    embeds = torch.cat([e1.hidden_states[-2], e2.hidden_states[-2]], dim=-1).half()
    caption_cache[cap] = (embeds.cpu(), e2.text_embeds.half().cpu())
print(f"  {len(caption_cache)} captions encoded")

latent_cache = []
for i, img_path in enumerate(imgs):
    pixels = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).half().to(DEVICE)
    with torch.no_grad():
        lat = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
    latent_cache.append(lat.squeeze(0).cpu())
    if (i+1) % 10 == 0:
        print(f"  Images: {i+1}/{len(imgs)}")

# Free everything except UNet
del vae, text_enc1, text_enc2, pipe
gc.collect()
torch.cuda.empty_cache()
print(f"  VRAM after freeing: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Apply LoRA — keep UNet in fp16 ────────────────────────
print("\n[4/5] Applying LoRA (fp16 UNet)...")
lora_cfg = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=["to_q","to_k","to_v","to_out.0"],
    lora_dropout=0.0, bias="none",
)
unet = get_peft_model(unet, lora_cfg)
# UNet stays fp16 — only LoRA params need grad
unet = unet.to(DEVICE)
unet.print_trainable_parameters()
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Cast only LoRA parameters to float32 for stable gradients
for name, param in unet.named_parameters():
    if param.requires_grad:
        param.data = param.data.float()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet.parameters()),
    lr=LEARNING_RATE, weight_decay=1e-2,
)
lr_scheduler = get_scheduler(
    "cosine", optimizer=optimizer,
    num_warmup_steps=30, num_training_steps=TRAIN_STEPS,
)
scaler = torch.cuda.amp.GradScaler()

# ── Train ─────────────────────────────────────────────────
print(f"\n[5/5] Training {TRAIN_STEPS} steps...")
print("-" * 55)

unet.train()
global_step, losses, skipped = 0, [], 0
optimizer.zero_grad()
pairs = list(zip(latent_cache, caps))

while global_step < TRAIN_STEPS:
    random.shuffle(pairs)
    for latent, cap in pairs:
        if global_step >= TRAIN_STEPS: break

        latents = latent.unsqueeze(0).to(DEVICE, dtype=torch.float16)
        embeds, pooled = caption_cache[cap]
        embeds  = embeds.to(DEVICE)
        pooled  = pooled.to(DEVICE)

        noise     = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_sch.config.num_train_timesteps,
                                  (1,), device=DEVICE).long()
        noisy     = noise_sch.add_noise(latents, noise, timesteps)
        add_time  = torch.tensor(
            [[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]],
            dtype=torch.float16, device=DEVICE
        )

        try:
            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(
                    noisy, timesteps,
                    encoder_hidden_states=embeds,
                    added_cond_kwargs={"time_ids": add_time, "text_embeds": pooled},
                ).sample
            # Loss in float32 to prevent NaN
            loss = F.mse_loss(pred.float(), noise.float())
        except Exception as e:
            skipped += 1
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            global_step += 1
            continue

        if torch.isnan(loss) or torch.isinf(loss):
            skipped += 1
            optimizer.zero_grad()
            global_step += 1
            continue

        scaler.scale(loss / GRAD_ACCUM).backward()

        if (global_step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in unet.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        global_step += 1

        if global_step % 25 == 0:
            avg  = sum(losses[-25:]) / max(len(losses[-25:]), 1)
            vram = torch.cuda.memory_allocated()/1e9
            print(f"  Step {global_step:>3}/{TRAIN_STEPS} | loss: {avg:.4f} | skipped: {skipped} | vram: {vram:.1f}GB")

# ── Save ──────────────────────────────────────────────────
print("\n  Saving LoRA weights...")
unet.save_pretrained(OUTPUT_DIR)
valid = [l for l in losses if l == l and l < 10]
final = round(sum(valid[-50:]) / max(len(valid[-50:]), 1), 4) if valid else 0.0
with open(os.path.join(OUTPUT_DIR, "lora_metadata.json"), "w") as f:
    json.dump({"style":"minimalist_luxury_streetwear","trigger":"lux_street",
               "steps":global_step,"rank":LORA_RANK,"final_loss":final,
               "skipped":skipped}, f, indent=2)

print()
print("=" * 55)
print(f"  LoRA training complete!")
print(f"  Valid steps    : {len(valid)}")
print(f"  Skipped steps  : {skipped}")
print(f"  Final avg loss : {final}")
print(f"  Saved to       : {OUTPUT_DIR}")
print("=" * 55)
