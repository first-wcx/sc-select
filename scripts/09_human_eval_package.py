from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", required=True, type=Path, help="CSV with prompt_id,prompt,method,image_id,image_path")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / "annotation_tasks.jsonl"
    template_path = args.output_dir / "annotation_template.csv"
    readme_path = args.output_dir / "README_for_annotators.md"

    grouped: dict[str, dict] = {}
    with args.selections.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt_id = row["prompt_id"]
            item = grouped.setdefault(prompt_id, {
                "prompt_id": prompt_id,
                "prompt": row["prompt"],
                "candidates": [],
            })
            item["candidates"].append({
                "candidate_id": f"{prompt_id}_{len(item['candidates'])}",
                "image_id": row["image_id"],
                "image_path": row["image_path"],
            })

    with tasks_path.open("w", encoding="utf-8") as handle:
        for item in grouped.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    with template_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["annotator_id", "prompt_id", "best_candidate_id", "tie_candidate_ids", "violated_constraints", "notes"])
        for prompt_id in grouped:
            writer.writerow(["", prompt_id, "", "", "", ""])

    readme_path.write_text(
        "# Annotation Instructions\n\n"
        "Choose the candidate image that best follows the prompt. Focus on object presence, attribute binding, relations, counting, and readable text. "
        "If multiple candidates are equally good, list ties in `tie_candidate_ids`. Do not judge only visual beauty.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

