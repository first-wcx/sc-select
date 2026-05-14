from __future__ import annotations

import argparse
import csv
import html
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["prompt_id"]] = row
    return rows


def rel_image_path(dataset: str, method: str, row: dict) -> str:
    src = Path(row["image_path"])
    return str(Path("images") / dataset / method / row["prompt_id"] / src.name).replace("\\", "/")


def build_tasks(selection_dir: Path, dataset: str, baselines: list[str], per_baseline: int, seed: int) -> tuple[list[dict], list[str]]:
    rng = random.Random(seed)
    csr_rows = read_jsonl(selection_dir / "csr_select_selected.jsonl")
    tasks: list[dict] = []
    manifest: list[str] = []

    for baseline in baselines:
        baseline_path = selection_dir / f"{baseline}_selected.jsonl"
        if not baseline_path.exists():
            continue
        base_rows = read_jsonl(baseline_path)
        common = sorted(set(csr_rows) & set(base_rows))
        different = [
            pid for pid in common
            if csr_rows[pid].get("image_path") != base_rows[pid].get("image_path")
        ]
        rng.shuffle(different)
        for pid in different[:per_baseline]:
            csr = csr_rows[pid]
            base = base_rows[pid]
            pair = [
                ("csr_select", csr, rel_image_path(dataset, "csr_select", csr)),
                (baseline, base, rel_image_path(dataset, baseline, base)),
            ]
            rng.shuffle(pair)
            task_id = f"{dataset}_{baseline}_{pid}"
            tasks.append({
                "task_id": task_id,
                "dataset": dataset,
                "prompt_id": pid,
                "prompt": csr["prompt"],
                "baseline": baseline,
                "left_image": pair[0][2],
                "right_image": pair[1][2],
                "left_method": pair[0][0],
                "right_method": pair[1][0],
                "left_source_path": pair[0][1]["image_path"],
                "right_source_path": pair[1][1]["image_path"],
            })
            manifest.extend([pair[0][1]["image_path"], pair[1][1]["image_path"]])
    return tasks, sorted(set(manifest))


def write_html(tasks: list[dict], output_path: Path) -> None:
    rows = []
    for t in tasks:
        rows.append(
            "<section class='task'>"
            f"<h2>{html.escape(t['task_id'])}</h2>"
            f"<p class='prompt'>{html.escape(t['prompt'])}</p>"
            "<div class='pair'>"
            f"<figure><img src='{html.escape(t['left_image'])}'><figcaption>Left</figcaption></figure>"
            f"<figure><img src='{html.escape(t['right_image'])}'><figcaption>Right</figcaption></figure>"
            "</div>"
            "</section>"
        )
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>SC-Select Human Evaluation</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f6f7f9;color:#15171a}"
        ".task{background:white;border:1px solid #d7dce2;border-radius:8px;padding:16px;margin:0 0 18px}"
        "h2{font-size:16px;margin:0 0 8px}.prompt{font-size:15px;margin:0 0 14px}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}"
        "img{width:100%;max-height:520px;object-fit:contain;background:#eef0f3;border:1px solid #ccd2da}"
        "figcaption{text-align:center;font-weight:700;margin-top:6px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>"
        "</head><body><h1>SC-Select Human Evaluation</h1>"
        "<p>For each prompt, choose the image that better satisfies object presence, attributes, relations, counting, and text constraints. Do not use visual beauty as the primary criterion.</p>"
        + "\n".join(rows)
        + "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--per-baseline", type=int, default=12)
    parser.add_argument("--self300-dir", type=Path, default=Path("outputs/selections/self300_sdxl_qwen_all_rewards_20260514"))
    parser.add_argument("--flux-dir", type=Path, default=Path("outputs/selections/flux_self300_stratified_60_all_rewards_20260514"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baselines = ["fixed_seed0", "clipscore_select", "image_reward_select", "pickscore_select", "hpsv2_select"]
    tasks: list[dict] = []
    manifest: list[str] = []
    for dataset, path in [("self300_sdxl", args.self300_dir), ("flux60", args.flux_dir)]:
        new_tasks, new_manifest = build_tasks(path, dataset, baselines, args.per_baseline, args.seed)
        tasks.extend(new_tasks)
        manifest.extend(new_manifest)

    tasks.sort(key=lambda r: r["task_id"])
    with (args.output_dir / "pairwise_tasks_blinded.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["task_id", "dataset", "prompt_id", "prompt", "left_image", "right_image"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in tasks:
            writer.writerow({k: row[k] for k in fields})

    with (args.output_dir / "pairwise_answer_key.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["task_id", "dataset", "prompt_id", "baseline", "left_method", "right_method", "left_source_path", "right_source_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in tasks:
            writer.writerow({k: row[k] for k in fields})

    with (args.output_dir / "annotation_response_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["annotator_id", "task_id", "choice", "tie", "bad_prompt_or_both_fail", "notes"])
        for row in tasks:
            writer.writerow(["", row["task_id"], "", "", "", ""])

    copy_rows: dict[tuple[str, str], dict] = {}
    for row in tasks:
        for side in ["left", "right"]:
            copy_rows[(row[f"{side}_source_path"], row[f"{side}_image"])] = {
                "source_path": row[f"{side}_source_path"],
                "local_relative_path": row[f"{side}_image"],
            }
    with (args.output_dir / "image_copy_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "local_relative_path"])
        writer.writeheader()
        for _, value in sorted(copy_rows.items()):
            writer.writerow(value)

    (args.output_dir / "remote_image_manifest.txt").write_text("\n".join(sorted(set(manifest))) + "\n", encoding="utf-8")
    write_html(tasks, args.output_dir / "annotation_viewer.html")
    (args.output_dir / "README.md").write_text(
        "# Human evaluation package\n\n"
        "- `annotation_viewer.html`: blinded side-by-side image viewer.\n"
        "- `annotation_response_template.csv`: annotator response sheet. `choice` must be `left`, `right`, or `tie`.\n"
        "- `pairwise_tasks_blinded.csv`: task metadata without method names.\n"
        "- `pairwise_answer_key.csv`: method mapping for analysis; do not give this file to annotators.\n"
        "- `image_copy_manifest.csv`: source-to-local image copy plan.\n"
        "- `remote_image_manifest.txt`: unique source image paths to copy from the remote project root.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(tasks)} pairwise tasks to {args.output_dir}")
    print(f"Unique remote images: {len(set(manifest))}")


if __name__ == "__main__":
    main()
