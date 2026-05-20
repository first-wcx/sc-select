#!/usr/bin/env python3
"""Qwen2.5-VL perception graph extraction for 300-prompt dataset.

Run AFTER SDXL generation completes.
Requires: transformers, qwen_vl_utils, torch, PIL
"""
import json
import torch
import time
import re
from pathlib import Path
from tqdm import tqdm

IMAGE_DIR = Path("outputs/images_300")
CONTRACT_FILE = Path("data/scselect_complex_300_contracts.jsonl")
OUTPUT_DIR = Path("outputs/perception_graphs_300")
LOG_FILE = Path("outputs/logs_300/qwen_parsing.log")

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log = open(LOG_FILE, "w", encoding="utf-8")

def logmsg(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

PARSE_PROMPT = """Analyze this image carefully and output a JSON object with these fields:
- "visible_objects": list of objects, each with "name" (string) and "attributes" (dict of attribute_name: value)
- "relations": list of {"subject": str, "relation": str, "object": str}
- "count": list of {"object": str, "number": int}
- "ocr_text": list of strings (any visible text/signs/labels)
- "uncertain": list of things you are unsure about

Output ONLY valid JSON, no explanation."""


def extract_json_from_response(text):
    """Try to extract JSON from model response."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try finding JSON block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"raw_response": text, "parse_error": "could not extract JSON"}


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    logmsg(f"GPU: {torch.cuda.get_device_name(0)}")
    logmsg(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load contracts
    contracts = {}
    with open(CONTRACT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                contracts[c["prompt_id"]] = c

    # Find all images
    image_files = sorted(IMAGE_DIR.glob("*/*.png"))
    logmsg(f"Total images: {len(image_files)}")

    # Check existing
    existing = list(OUTPUT_DIR.glob("*.json"))
    logmsg(f"Already parsed: {len(existing)}")
    existing_ids = {p.stem for p in existing}

    # Load model
    logmsg("Loading Qwen2.5-VL-7B...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.auto, device_map="auto"
    )
    logmsg("Model loaded.")

    start_time = time.time()
    parsed = 0
    errors = 0

    for img_path in tqdm(image_files, desc="Qwen parsing"):
        pid = img_path.parent.name
        img_id = img_path.stem
        out_name = f"{pid}_{img_id}"

        if out_name in existing_ids:
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
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=2048)

            generated = output_ids[0][inputs.input_ids.shape[1]:]
            response = processor.decode(generated, skip_special_tokens=True)

            # Parse response
            parsed_json = extract_json_from_response(response)
            parsed_json["prompt_id"] = pid
            parsed_json["image_id"] = img_id
            parsed_json["parser"] = "Qwen2.5-VL-7B"

            out_path = OUTPUT_DIR / f"{out_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)

            parsed += 1
        except Exception as e:
            logmsg(f"ERROR {out_name}: {e}")
            errors += 1

    elapsed = time.time() - start_time
    logmsg(f"Done. Parsed: {parsed}, Errors: {errors}")
    logmsg(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logmsg(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.close()


if __name__ == "__main__":
    main()
