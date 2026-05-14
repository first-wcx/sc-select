#!/usr/bin/env python3
"""Generate SDXL images for a JSONL prompt file."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else output_dir.parent / "logs" / "generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")

    def logmsg(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    prompts = load_jsonl(prompt_file)
    logmsg(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    logmsg(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logmsg(f"Prompt file: {prompt_file}")
    logmsg(f"Total prompts: {len(prompts)}, seeds: {seeds}")

    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    logmsg("SDXL loaded.")

    generated = 0
    errors = 0
    start_time = time.time()
    for item in tqdm(prompts, desc="Generating"):
        pid = item["prompt_id"]
        save_dir = output_dir / pid
        save_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            out_path = save_dir / f"seed_{seed}.png"
            if out_path.exists():
                continue
            try:
                generator = torch.Generator("cuda").manual_seed(seed)
                image = pipe(
                    item["prompt"],
                    generator=generator,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                ).images[0]
                image.save(out_path)
                generated += 1
            except Exception as exc:
                errors += 1
                logmsg(f"ERROR {pid} seed_{seed}: {exc}")

    elapsed = time.time() - start_time
    logmsg(f"Done. Generated: {generated}, Errors: {errors}")
    logmsg(f"Time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    logmsg(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.close()


if __name__ == "__main__":
    main()
