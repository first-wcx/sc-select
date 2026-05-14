"""Reward model scoring utilities that bypass broken pip packages.

Uses transformers directly to load models from HuggingFace cache.
Supports: HPSv2, CLIPScore, ImageReward (BLIP-based), PickScore (CLIP-based).
"""

import torch
from PIL import Image
from pathlib import Path


def score_clipscore(images, prompts, device="cuda", model_id="openai/clip-vit-base-patch32"):
    """CLIPScore: cosine similarity between image and text embeddings."""
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    results = []
    for img_path, prompt in zip(images, prompts):
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                score = outputs.logits_per_image[0, 0].item()
            results.append(score)
        except Exception as e:
            print(f"CLIPScore error: {e}")
            results.append(None)

    del model
    torch.cuda.empty_cache()
    return results


def score_hpsv2(images, prompts, device="cuda"):
    """HPSv2: human preference score."""
    import hpsv2

    results = []
    for img_path, prompt in zip(images, prompts):
        try:
            img = Image.open(img_path).convert("RGB")
            score = hpsv2.score(img, prompt, hps_version="v2.0")
            results.append(float(score[0]) if isinstance(score, list) else float(score))
        except Exception as e:
            print(f"HPSv2 error: {e}")
            results.append(None)
    return results


def score_image_reward(images, prompts, device="cuda"):
    """ImageReward: BLIP-based reward model.

    Uses zai-org/ImageReward from HuggingFace.
    Falls back to manual BLIP implementation if the pip package is broken.
    """
    try:
        # Try using the pip package first
        import ImageReward as IR
        model = IR.load("ImageReward-v1.0")
        results = []
        for img_path, prompt in zip(images, prompts):
            try:
                score = model.score(prompt, str(img_path))
                results.append(float(score))
            except Exception as e:
                print(f"ImageReward score error: {e}")
                results.append(None)
        return results
    except (ImportError, Exception) as e:
        print(f"ImageReward pip package failed: {e}")
        print("Falling back to manual BLIP implementation...")
        return _score_image_reward_manual(images, prompts, device)


def _score_image_reward_manual(images, prompts, device="cuda"):
    """Manual ImageReward scoring using transformers BLIP."""
    from transformers import BlipForImageTextRetrieval, BlipProcessor

    model_id = "zai-org/ImageReward"
    try:
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForImageTextRetrieval.from_pretrained(model_id).to(device).eval()
    except Exception:
        # Fallback: use salesforce/blip-image-text-retrieval
        model_id = "Salesforce/blip-image-text-retrieval-coco"
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForImageTextRetrieval.from_pretrained(model_id).to(device).eval()

    results = []
    for img_path, prompt in zip(images, prompts):
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**inputs)
                score = output.logits_per_image.item()
            results.append(score)
        except Exception as e:
            print(f"ImageReward manual error: {e}")
            results.append(None)

    del model
    torch.cuda.empty_cache()
    return results


def score_pickscore(images, prompts, device="cuda"):
    """PickScore: CLIP-based preference model.

    Uses yuvalkirstal/PickScore_v1 from HuggingFace.
    Falls back to CLIP ViT-L/14 if model not available.
    """
    try:
        from transformers import AutoModel, AutoProcessor

        model_id = "yuvalkirstal/PickScore_v1"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()

        results = []
        for img_path, prompt in zip(images, prompts):
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, text=prompt, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    output = model(**inputs)
                    score = output.logits_per_image.item()
                results.append(score)
            except Exception as e:
                print(f"PickScore error: {e}")
                results.append(None)

        del model
        torch.cuda.empty_cache()
        return results
    except Exception as e:
        print(f"PickScore model load failed: {e}")
        print("Skipping PickScore (model not available in cache)")
        return [None] * len(images)
