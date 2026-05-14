#!/usr/bin/env python3
"""Generic InternVL perception graph extraction for SC-Select images."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


PARSE_PROMPT = """Analyze this image carefully and output a JSON object with these fields:
- "visible_objects": list of objects, each with "name" (string) and "attributes" (dict of attribute_name: value)
- "relations": list of {"subject": str, "relation": str, "object": str}
- "count": list of {"object": str, "number": int}
- "ocr_text": list of strings (any visible text/signs/labels)
- "uncertain": list of things you are unsure about

Output ONLY valid JSON, no explanation."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_json_from_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"raw_response": text, "parse_error": "could not extract JSON"}


def build_transform(input_size: int):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int, max_num: int, image_size: int, use_thumbnail: bool) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(path: Path, input_size: int, max_num: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, min_num=1, max_num=max_num, image_size=input_size, use_thumbnail=True)
    return torch.stack([transform(img) for img in images])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default="OpenGVLab/InternVL2_5-4B")
    parser.add_argument("--parser-name", default="InternVL2.5-4B")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--limit-images", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(image_dir.glob("*/*.png"))
    if args.limit_images > 0:
        image_files = image_files[: args.limit_images]

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Image dir: {image_dir}", flush=True)
    print(f"Images: {len(image_files)}", flush=True)
    print(f"Already parsed: {len(list(out_dir.glob('*.json')))}", flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    print("Model loaded.", flush=True)

    parsed = 0
    errors = 0
    for img_path in tqdm(image_files, desc="InternVL parsing"):
        pid = img_path.parent.name
        img_id = img_path.stem
        out_path = out_dir / f"{pid}_{img_id}.json"
        if out_path.exists():
            continue
        try:
            pixel_values = load_image(img_path, input_size=448, max_num=args.max_tiles).to(dtype)
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            response = model.chat(tokenizer, pixel_values, PARSE_PROMPT, generation_config)
            parsed_json = extract_json_from_response(response)
            parsed_json["prompt_id"] = pid
            parsed_json["image_id"] = img_id
            parsed_json["parser"] = args.parser_name
            parsed_json["image_path"] = str(img_path)
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(parsed_json, handle, ensure_ascii=False, indent=2)
            parsed += 1
        except Exception as exc:
            errors += 1
            print(f"ERROR {pid}_{img_id}: {exc}", flush=True)

    print(f"Done. Parsed: {parsed}, Errors: {errors}", flush=True)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
