import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, torch, warnings, json, random, gc, time
warnings.filterwarnings("ignore")

# ── ENV SETUP ──────────────────────────────────────────────────────────────
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"]  = r"E:\Programe Files\huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

print("=" * 60)
print("  Week 4 — Brand Style LoRA Training Pipeline (bfloat16)")
print("  Target Aesthetic: Minimalist Luxury Streetwear (lux_street)")
print("=" * 60)

DATASET_DIR = r"E:\Fashion model\data\lora_dataset\minimalist_streetwear"
OUTPUT_DIR  = r"E:\Fashion model\models\lora_lux_street"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESOLUTION    = 512
TRAIN_STEPS   = 300
LEARNING_RATE = 1e-4
GRAD_ACCUM    = 4
LORA_RANK     = 2
LORA_ALPHA    = 4
DEVICE        = "cuda"
DTYPE         = torch.bfloat16  # <-- The Magic Fix for NaN!

# ── PHASE 1: DATASET INGESTION ──────────────────────────────────────────────
print("\n[1/6] Loading fashion image-caption pairs...")
from torchvision import transforms
from PIL import Image

imgs, caps = [], []
for f in sorted(os.listdir(DATASET_DIR)):
    if not f.endswith(".jpg"): continue
    txt = os.path.join(DATASET_DIR, f.replace(".jpg", ".txt"))
    if not os.path.exists(txt): continue
    imgs.append(os.path.join(DATASET_DIR, f))
    with open(txt, "r") as fp: 
        caps.append(fp.read().strip())

tf = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
print(f"  ✓ Found {len(imgs)} images with corresponding .txt captions.")

# ── PHASE 2: PIPELINE INFLATION ──────────────────────────────────────────────
print("\n[2/6] Inflating SDXL base pipeline components...")
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16", # Loads the fp16 weights, but casts directly to bfloat16 in memory
    local_files_only=True,
)

tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
text_enc1, text_enc2 = pipe.text_encoder, pipe.text_encoder_2
vae, unet = pipe.vae.to(DEVICE), pipe.unet

noise_sch = DDPMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="scheduler", 
    local_files_only=True,
)

text_enc1.requires_grad_(False)
text_enc2.requires_grad_(False)
vae.requires_grad_(False)

# ── PHASE 3: COMPUTE ENGAGEMENT & STREAMING ─────────────────────────────────
print("\n[3/6] Pre-encoding tokens & caching latents to System RAM...")
caption_cache = {}
for cap in set(caps):
    t1 = tok1([cap], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    t2 = tok2([cap], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.no_grad():
        e1 = text_enc1(t1.input_ids, output_hidden_states=True)
        e2 = text_enc2(t2.input_ids, output_hidden_states=True)
    embeds = torch.cat([e1.hidden_states[-2], e2.hidden_states[-2]], dim=-1).to(dtype=DTYPE)
    caption_cache[cap] = (embeds.cpu(), e2.text_embeds.to(dtype=DTYPE).cpu())

latent_cache = []
for i, img_path in enumerate(imgs):
    pixels = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        lat = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
    latent_cache.append(lat.squeeze(0).cpu())
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(imgs)} samples...")

# ── PHASE 4: THE MEMORY PURGE ───────────────────────────────────────────────
print("\n[4/6] Purging heavy encoder architectures from GPU VRAM...")
del vae, text_enc1, text_enc2, pipe
gc.collect()
torch.cuda.empty_cache()

lora_cfg = LoraConfig(
    r=LORA_RANK, 
    lora_alpha=LORA_ALPHA,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0, 
    bias="none",
)
unet = get_peft_model(unet, lora_cfg).to(DEVICE)
unet.enable_gradient_checkpointing()

# Lock native weights to bfloat16; ensure only LoRA uses float32
for name, param in unet.named_parameters():
    if param.requires_grad:
        param.data = param.data.float()

unet.print_trainable_parameters()
print(f"  Current baseline operational VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── PHASE 5: ISOLATION RUNTIME TEST ──────────────────────────────────────────
print("\n[5/6] Verifying isolation runtime forward pass...")
test_lat = latent_cache[0].unsqueeze(0).to(DEVICE, dtype=DTYPE)
test_e, test_p = caption_cache[caps[0]]
test_noise = torch.randn_like(test_lat)
test_ts    = torch.tensor([500], device=DEVICE)
test_noisy = noise_sch.add_noise(test_lat, test_noise, test_ts)
test_time  = torch.tensor([[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]], dtype=DTYPE, device=DEVICE)

with torch.no_grad():
    with torch.autocast("cuda", dtype=DTYPE):
        test_out = unet(
            test_noisy, test_ts,
            encoder_hidden_states=test_e.to(DEVICE),
            added_cond_kwargs={"time_ids": test_time, "text_embeds": test_p.to(DEVICE)},
        ).sample

print(f"  -> Test Matrix Shape : {test_out.shape}")
print(f"  -> Numerical Overflow Check (NaN?): {torch.isnan(test_out).any().item()}")
if torch.isnan(test_out).any():
    print("  [CRITICAL ALERT] Numeric variance error. Exiting before train loop.")
    sys.exit()
print("  ✓ Forward pass completely stable.")

# ── PHASE 6: EXECUTE OPTIMIZATION LOOP ───────────────────────────────────────
print(f"\n[6/6] Launching structural optimization ({TRAIN_STEPS} steps)...")
print("-" * 60)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=LEARNING_RATE, weight_decay=1e-2)
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=30, num_training_steps=TRAIN_STEPS)

unet.train()
global_step, losses = 0, []
optimizer.zero_grad()
pairs = list(zip(latent_cache, caps))

while global_step < TRAIN_STEPS:
    random.shuffle(pairs)
    for latent, cap in pairs:
        if global_step >= TRAIN_STEPS: break
        
        latents = latent.unsqueeze(0).to(DEVICE, dtype=DTYPE)
        embeds, pooled = caption_cache[cap]
        embeds, pooled = embeds.to(DEVICE), pooled.to(DEVICE)
        
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_sch.config.num_train_timesteps, (1,), device=DEVICE).long()
        noisy = noise_sch.add_noise(latents, noise, timesteps)
        add_time = torch.tensor([[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]], dtype=DTYPE, device=DEVICE)
        
        with torch.autocast("cuda", dtype=DTYPE):
            pred = unet(noisy, timesteps, encoder_hidden_states=embeds, added_cond_kwargs={"time_ids": add_time, "text_embeds": pooled}).sample
            loss = F.mse_loss(pred.float(), noise.float())
            
        # Standard backward pass without GradScaler 
        loss = loss / GRAD_ACCUM
        loss.backward()
        
        if (global_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        losses.append(loss.item() * GRAD_ACCUM)
        global_step += 1
        
        if global_step % 25 == 0:
            avg_loss = sum(losses[-25:]) / 25
            print(f"  Step {global_step:>3}/{TRAIN_STEPS} | Rolling Loss: {avg_loss:.4f} | VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Save out standalone weights
unet.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "lora_metadata.json"), "w") as f:
    json.dump({
        "style": "minimalist_luxury_streetwear",
        "trigger": "lux_street",
        "steps": global_step,
        "final_loss": round(sum(losses[-25:]) / 25, 4)
    }, f, indent=2)

print("\n" + "=" * 60)
print("  ✓ SUCCESS: LoRA Training Cycle Complete!")
print(f"  ✓ Dynamic configuration states exported to: {OUTPUT_DIR}")
print("=" * 60)