#!/usr/bin/env python3
"""Add CLIPScore values to a candidate score CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    with Path(args.scores).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no rows")

    processor = CLIPProcessor.from_pretrained(args.model_id)
    model = CLIPModel.from_pretrained(args.model_id).to(device).eval()

    for row in tqdm(rows, desc="CLIPScore"):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = project_root / image_path
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                text=[row["prompt"]],
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            row["clipscore"] = outputs.logits_per_image[0, 0].item()
        except Exception as exc:
            print(f"CLIPScore error for {image_path}: {exc}", flush=True)
            row["clipscore"] = ""

    fieldnames = list(rows[0].keys())
    if "clipscore" not in fieldnames:
        fieldnames.append("clipscore")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
