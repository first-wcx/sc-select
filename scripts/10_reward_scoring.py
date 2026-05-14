#!/usr/bin/env python3
"""Score candidate images with reward models: HPSv2, ImageReward, PickScore.

Usage:
    python scripts/10_reward_scoring.py --dataset self50 --server local
    python scripts/10_reward_scoring.py --dataset geneval --server local

Requires:
    - hpsv2 (pip install hpsv2)
    - transformers (for BLIP-based ImageReward and CLIP-based PickScore)
    - torch, PIL, pandas
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


# ---- HPSv2 ----

def score_hpsv2(images, prompts, device="cuda"):
    """Score images using HPSv2."""
    import hpsv2.img_score as hpsv2
    results = []
    for img_path, prompt in tqdm(zip(images, prompts), total=len(images), desc="HPSv2"):
        try:
            cp_path = os.environ.get("HPSV2_CHECKPOINT")
            if cp_path:
                score = hpsv2.score(img_path, prompt, cp=cp_path, hps_version="v2.0")
            else:
                score = hpsv2.score(img_path, prompt, hps_version="v2.0")
            if isinstance(score, (list, tuple)):
                score = score[0]
            results.append(float(score))
        except Exception as e:
            print(f"HPSv2 error for {img_path}: {e}")
            results.append(None)
    return results


# ---- ImageReward (BLIP-based) ----

def score_image_reward(images, prompts, device="cuda"):
    """Score images using the ImageReward package."""
    import torch
    import transformers.modeling_utils as modeling_utils
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        heads = set(heads) - already_pruned_heads
        mask = torch.ones(n_heads, head_size)
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    # ImageReward imports compatibility helpers from older transformers locations.
    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
        modeling_utils.PreTrainedModel.all_tied_weights_keys = property(
            lambda self: {}
        )
    if not hasattr(modeling_utils.PreTrainedModel, "_convert_head_mask_to_5d"):
        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            dtype = next(self.parameters()).dtype
            return head_mask.to(dtype=dtype)

        modeling_utils.PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
    if not hasattr(modeling_utils.PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        modeling_utils.PreTrainedModel.get_head_mask = get_head_mask

    import ImageReward as IR
    from transformers import BertTokenizer
    import ImageReward.models.BLIP.blip as blip_module
    import ImageReward.models.BLIP.blip_pretrain as blip_pretrain_module

    def init_tokenizer_compat():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        return tokenizer

    blip_module.init_tokenizer = init_tokenizer_compat
    blip_pretrain_module.init_tokenizer = init_tokenizer_compat

    download_root = os.environ.get("IMAGE_REWARD_DOWNLOAD_ROOT", "/root/autodl-tmp/ImageReward_cache")
    print(f"Loading ImageReward-v1.0 from {download_root}...")
    model = IR.load("ImageReward-v1.0", device=device, download_root=download_root).eval()

    results = []
    for img_path, prompt in tqdm(zip(images, prompts), total=len(images), desc="ImageReward"):
        try:
            with torch.no_grad():
                score = model.score(prompt, img_path)
            results.append(float(score))
        except Exception as e:
            print(f"ImageReward error for {img_path}: {e}")
            results.append(None)

    del model
    torch.cuda.empty_cache()
    return results


# ---- PickScore (CLIP-based) ----

def score_pickscore(images, prompts, device="cuda"):
    """Score images using PickScore model (CLIP-based)."""
    from transformers import AutoModel, CLIPProcessor

    processor_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id = "yuvalkirstain/PickScore_v1"
    print(f"Loading PickScore processor from {processor_id}...")
    processor = CLIPProcessor.from_pretrained(processor_id)
    print(f"Loading PickScore model from {model_id}...")
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    results = []
    for img_path, prompt in tqdm(zip(images, prompts), total=len(images), desc="PickScore"):
        try:
            img = load_image(img_path)
            image_inputs = processor(
                images=img,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            text_inputs = processor(
                text=prompt,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                image_emb = model.get_image_features(**image_inputs)
                if hasattr(image_emb, "pooler_output"):
                    image_emb = image_emb.pooler_output
                image_emb = image_emb / torch.norm(image_emb, dim=-1, keepdim=True)
                text_emb = model.get_text_features(**text_inputs)
                if hasattr(text_emb, "pooler_output"):
                    text_emb = text_emb.pooler_output
                text_emb = text_emb / torch.norm(text_emb, dim=-1, keepdim=True)
                score = model.logit_scale.exp() * (text_emb @ image_emb.T)[0, 0]
            results.append(float(score))
        except Exception as e:
            print(f"PickScore error for {img_path}: {e}")
            results.append(None)

    del model
    torch.cuda.empty_cache()
    return results


# ---- CLIPScore ----

def score_clipscore(images, prompts, device="cuda"):
    """Score images using CLIP ViT-B/32."""
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model from {model_id}...")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    results = []
    for img_path, prompt in tqdm(zip(images, prompts), total=len(images), desc="CLIPScore"):
        try:
            img = load_image(img_path)
            inputs = processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                score = outputs.logits_per_image[0, 0].item()
            results.append(score)
        except Exception as e:
            print(f"CLIPScore error for {img_path}: {e}")
            results.append(None)

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["self50", "self300", "geneval"])
    parser.add_argument("--metrics", nargs="+", default=["hpsv2", "clipscore"],
                        choices=["hpsv2", "image_reward", "pickscore", "clipscore"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project-root", default="/root/autodl-tmp/projects/SCAlign")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only score first 5 images")
    args = parser.parse_args()

    project = Path(args.project_root)
    out_dir = Path(args.output_dir) if args.output_dir else project / "outputs" / "reward_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSR results to get image paths and prompts
    if args.dataset == "self50":
        csr_file = project / "outputs" / "csr_scores" / "csr_results.jsonl"
        prompt_file = project / "data" / "raw_prompts_50.jsonl"
    elif args.dataset == "self300":
        csr_file = project / "outputs" / "csr_scores_300" / "csr_results.jsonl"
        prompt_file = project / "data" / "scselect_complex_300.jsonl"
    elif args.dataset == "geneval":
        csr_file = project / "outputs" / "geneval" / "csr_scores" / "csr_results.jsonl"
        prompt_file = project / "data" / "geneval" / "geneval_prompts.jsonl"

    if not csr_file.exists():
        print(f"CSR file not found: {csr_file}")
        return

    # Load data
    prompts_map = {}
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                prompts_map[d["prompt_id"]] = d["prompt"]

    csr_rows = []
    with open(csr_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                csr_rows.append(json.loads(line))

    # Build image path list
    image_dir = project / "outputs" / "images" if args.dataset == "self50" else project / "outputs" / "geneval" / "images"
    if args.dataset == "self300":
        image_dir = project / "outputs" / "images_300"
    if args.dataset == "geneval":
        image_dir = project / "outputs" / "geneval" / "images"

    images = []
    prompts = []
    prompt_ids = []
    image_ids = []

    for row in csr_rows:
        pid = row["prompt_id"]
        img_id = row.get("image_id", "")
        # Use image_path from CSR results if available
        rel_path = row.get("image_path", "")
        if rel_path:
            img_path = project / rel_path
        else:
            # Fallback: construct path from image_dir and prompt_id/image_id
            img_path = image_dir / pid / f"{img_id}.png"
        if not img_path.exists():
            # Try other patterns
            img_path = image_dir / f"{img_id}.png"
        if not img_path.exists():
            img_path = image_dir / pid / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        if pid in prompts_map:
            images.append(str(img_path))
            prompts.append(prompts_map[pid])
            prompt_ids.append(pid)
            image_ids.append(img_id)

    if args.dry_run:
        images = images[:5]
        prompts = prompts[:5]
        prompt_ids = prompt_ids[:5]
        image_ids = image_ids[:5]

    print(f"Scoring {len(images)} images with metrics: {args.metrics}")

    # Score with each metric
    all_scores = {"prompt_id": prompt_ids, "image_id": image_ids}

    for metric in args.metrics:
        print(f"\n=== {metric} ===")
        if metric == "hpsv2":
            scores = score_hpsv2(images, prompts, args.device)
        elif metric == "image_reward":
            scores = score_image_reward(images, prompts, args.device)
        elif metric == "pickscore":
            scores = score_pickscore(images, prompts, args.device)
        elif metric == "clipscore":
            scores = score_clipscore(images, prompts, args.device)
        else:
            continue
        all_scores[metric] = scores

    # Save results
    df = pd.DataFrame(all_scores)
    out_file = out_dir / f"{args.dataset}_reward_scores.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved to {out_file}")

    # Also save as JSONL
    out_jsonl = out_dir / f"{args.dataset}_reward_scores.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(len(prompt_ids)):
            row = {"prompt_id": prompt_ids[i], "image_id": image_ids[i]}
            for metric in args.metrics:
                row[metric] = all_scores[metric][i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved to {out_jsonl}")


if __name__ == "__main__":
    main()
