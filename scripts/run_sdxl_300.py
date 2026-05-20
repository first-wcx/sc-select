#!/usr/bin/env python3
"""Generate 1200 SDXL images for 300 prompts (4 seeds each)."""
import json
import torch
import time
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
from pathlib import Path

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
INPUT = Path("data/scselect_complex_300.jsonl")
OUTPUT_DIR = Path("outputs/images_300")
SEEDS = [0, 1, 2, 3]
LOG_FILE = Path("outputs/logs_300/generation.log")

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log = open(LOG_FILE, "w", encoding="utf-8")

def logmsg(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

logmsg(f"GPU: {torch.cuda.get_device_name(0)}")
logmsg(f"CUDA available: {torch.cuda.is_available()}")
logmsg(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

logmsg("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, use_safetensors=True
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
logmsg("SDXL loaded.")

prompts = []
with INPUT.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            prompts.append(json.loads(line))

logmsg(f"Total prompts: {len(prompts)}, seeds per prompt: {len(SEEDS)}")

existing = 0
for item in prompts:
    pid = item["prompt_id"]
    for seed in SEEDS:
        out_path = OUTPUT_DIR / pid / f"seed_{seed}.png"
        if out_path.exists():
            existing += 1
logmsg(f"Already existing: {existing}")

start_time = time.time()
generated = 0
errors = 0

for item in tqdm(prompts, desc="Generating"):
    pid = item["prompt_id"]
    prompt = item["prompt"]
    save_dir = OUTPUT_DIR / pid
    save_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        out_path = save_dir / f"seed_{seed}.png"
        if out_path.exists():
            continue
        try:
            generator = torch.Generator("cuda").manual_seed(seed)
            img = pipe(
                prompt,
                generator=generator,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
            img.save(out_path)
            generated += 1
        except Exception as e:
            logmsg(f"ERROR {pid} seed_{seed}: {e}")
            errors += 1

elapsed = time.time() - start_time
logmsg(f"Done. Generated: {generated}, Errors: {errors}")
logmsg(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
if generated > 0:
    logmsg(f"Avg per image: {elapsed/generated:.1f}s")
logmsg(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log.close()
