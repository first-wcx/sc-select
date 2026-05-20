#!/bin/bash
# Full pipeline for 300-prompt SC-Select experiment
# Run on server_004: bash run_300_full_pipeline.sh
#
# This script runs the complete pipeline:
# 1. SDXL image generation (300 prompts x 4 seeds = 1200 images)
# 2. Qwen2.5-VL perception graph extraction
# 3. CSR scoring and selection
# 4. CLIPScore baseline
# 5. Reward model baselines (HPSv2, ImageReward, PickScore)
# 6. Statistical significance tests
#
# Estimated time: 4-8 hours on RTX 5090
# Estimated disk: ~10-15GB for images + perception graphs

set -e
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=0
PYTHON=/root/miniconda3/bin/python
PROJECT=/root/autodl-tmp/projects/SCAlign
cd $PROJECT

echo "=== Step 0: Environment check ==="
$PYTHON -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.cuda.is_available())"
df -h /root/autodl-tmp | tail -1

echo ""
echo "=== Step 1: SDXL Generation (300 prompts x 4 seeds) ==="
# Generate 1200 images
$PYTHON -c "
import json, torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
from pathlib import Path

MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'
INPUT = Path('data/scselect_complex_300.jsonl')
OUTPUT_DIR = Path('outputs/images_300')
SEEDS = [0, 1, 2, 3]

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to('cuda')
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

prompts = []
with INPUT.open('r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            prompts.append(json.loads(line))

print(f'Generating {len(prompts)} prompts x {len(SEEDS)} seeds = {len(prompts)*len(SEEDS)} images')
for item in tqdm(prompts, desc='Generating'):
    pid = item['prompt_id']
    prompt = item['prompt']
    save_dir = OUTPUT_DIR / pid
    save_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        out_path = save_dir / f'seed_{seed}.png'
        if out_path.exists():
            continue
        generator = torch.Generator('cuda').manual_seed(seed)
        img = pipe(prompt, generator=generator, num_inference_steps=30, guidance_scale=7.5).images[0]
        img.save(out_path)
print('SDXL generation done.')
"

echo ""
echo "=== Step 2: Qwen2.5-VL Perception Graph Extraction ==="
$PYTHON -c "
import json, torch
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
IMAGE_DIR = Path('outputs/images_300')
CONTRACT_FILE = Path('data/scselect_complex_300_contracts.jsonl')
OUTPUT_DIR = Path('outputs/perception_graphs_300')

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.auto, device_map='auto'
)

contracts = {}
with CONTRACT_FILE.open('r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            contracts[c['prompt_id']] = c

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

prompt_dirs = sorted([d for d in IMAGE_DIR.iterdir() if d.is_dir()])
print(f'Processing {len(prompt_dirs)} prompt directories')

for prompt_dir in tqdm(prompt_dirs, desc='Qwen parsing'):
    pid = prompt_dir.name
    contract = contracts.get(pid, {})
    for img_path in sorted(prompt_dir.glob('*.png')):
        img_id = img_path.stem
        out_path = OUTPUT_DIR / f'{pid}_{img_id}.json'
        if out_path.exists():
            continue
        # Parse image with Qwen
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': str(img_path)}, {'type': 'text', 'text': 'Describe this image in detail. List all visible objects with their attributes (color, material, size), spatial relations between objects, count of each object type, and any visible text. Output as JSON with keys: objects, attributes, relations, count, ocr_text.'}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True)
        result = {'prompt_id': pid, 'image_id': img_id, 'raw_response': response, 'parser': 'Qwen2.5-VL-7B'}
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
print('Qwen parsing done.')
"

echo ""
echo "=== Step 3: CSR Scoring ==="
$PYTHON -c "
import json, sys
sys.path.insert(0, '.')
from src.scoring.csr import score_all
from pathlib import Path

CONTRACT_FILE = Path('data/scselect_complex_300_contracts.jsonl')
PERCEPTION_DIR = Path('outputs/perception_graphs_300')
OUTPUT = Path('outputs/csr_scores_300/csr_results.jsonl')
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

contracts = {}
with CONTRACT_FILE.open('r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            contracts[c['prompt_id']] = c

results = []
for pg_file in sorted(PERCEPTION_DIR.glob('*.json')):
    with pg_file.open('r', encoding='utf-8') as f:
        pg = json.load(f)
    pid = pg['prompt_id']
    contract = contracts.get(pid)
    if not contract:
        continue
    scores = score_all(contract, pg)
    scores['prompt_id'] = pid
    scores['image_id'] = pg['image_id']
    results.append(scores)

with OUTPUT.open('w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'CSR scoring done: {len(results)} results')
"

echo ""
echo "=== Step 4: CLIPScore Baseline ==="
$PYTHON scripts/10_reward_scoring.py --dataset self300 --metrics clipscore --project-root $PROJECT

echo ""
echo "=== Step 5: Reward Model Baselines ==="
$PYTHON scripts/10_reward_scoring.py --dataset self300 --metrics hpsv2 image_reward pickscore --project-root $PROJECT

echo ""
echo "=== Step 6: Baseline Selection ==="
$PYTHON scripts/07_run_baselines.py --dataset self300 --project-root $PROJECT

echo ""
echo "=== Step 7: Statistical Significance ==="
$PYTHON scripts/08_bootstrap_stats.py --dataset self300 --project-root $PROJECT

echo ""
echo "=== Pipeline Complete ==="
echo "Results in: outputs/"
