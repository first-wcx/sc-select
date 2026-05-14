#!/usr/bin/env python3
"""Append reward-model scores to an arbitrary candidate score CSV."""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path


def load_reward_module():
    script_path = Path(__file__).resolve().parent / "10_reward_scoring.py"
    spec = importlib.util.spec_from_file_location("reward_scoring", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metrics", nargs="+", required=True, choices=["hpsv2", "image_reward", "pickscore", "clipscore"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    with Path(args.scores).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no candidate rows")

    images = []
    prompts = []
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = project_root / image_path
        images.append(str(image_path))
        prompts.append(row["prompt"])

    reward = load_reward_module()
    metric_to_func = {
        "hpsv2": reward.score_hpsv2,
        "image_reward": reward.score_image_reward,
        "pickscore": reward.score_pickscore,
        "clipscore": reward.score_clipscore,
    }

    fieldnames = list(rows[0].keys())
    for metric in args.metrics:
        if metric not in fieldnames:
            fieldnames.append(metric)

    out_path = Path(args.output)
    for metric in args.metrics:
        print(f"=== {metric} ===", flush=True)
        scores = metric_to_func[metric](images, prompts, args.device)
        for row, score in zip(rows, scores):
            row[metric] = "" if score is None else score
        write_rows(out_path, rows, fieldnames)
        print(f"Checkpoint saved after {metric}: {out_path}", flush=True)

    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
