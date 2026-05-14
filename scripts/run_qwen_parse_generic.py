#!/usr/bin/env python3
"""Generic Qwen2.5-VL perception graph extraction for SC-Select images."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm


PARSE_PROMPT = """Analyze this image carefully and output a JSON object with these fields:
- "visible_objects": list of objects, each with "name" (string) and "attributes" (dict of attribute_name: value)
- "relations": list of {"subject": str, "relation": str, "object": str}
- "count": list of {"object": str, "number": int}
- "ocr_text": list of strings (any visible text/signs/labels)
- "uncertain": list of things you are unsure about

Output ONLY valid JSON, no explanation."""


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--parser-name", default="Qwen2.5-VL-7B")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit-images", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

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

    processor = AutoProcessor.from_pretrained(args.model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto"
    )
    print("Model loaded.", flush=True)

    parsed = 0
    errors = 0
    start = time.time()
    for img_path in tqdm(image_files, desc="Qwen parsing"):
        pid = img_path.parent.name
        img_id = img_path.stem
        out_path = out_dir / f"{pid}_{img_id}.json"
        if out_path.exists():
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": PARSE_PROMPT},
            ],
        }]

        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            generated = output_ids[0][inputs.input_ids.shape[1] :]
            response = processor.decode(generated, skip_special_tokens=True)
            parsed_json = extract_json_from_response(response)
            parsed_json["prompt_id"] = pid
            parsed_json["image_id"] = img_id
            parsed_json["parser"] = args.parser_name
            parsed_json["image_path"] = str(img_path)

            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(parsed_json, handle, ensure_ascii=False, indent=2)
            parsed += 1
        except Exception as exc:
            print(f"ERROR {pid}_{img_id}: {exc}", flush=True)
            errors += 1

    elapsed = time.time() - start
    print(f"Done. Parsed: {parsed}, Errors: {errors}", flush=True)
    print(f"Time: {elapsed:.0f}s ({elapsed / 60:.1f}min)", flush=True)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
