from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import (  # noqa: E402
    clipscore_select,
    csr_select,
    fixed_seed_select,
    hpsv2_select,
    image_reward_select,
    pickscore_select,
    random_select,
    vlm_direct_select,
)

SCORE_KEYS = ["csr_obj", "csr_attr", "csr_rel", "csr_count", "csr_text", "csr_all"]


def clean_csv_value(value: str) -> Any:
    if value == "":
        return None
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [{key: clean_csv_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def group_by_prompt(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)
    return dict(grouped)


def summarize(method: str, selected: list[dict[str, Any]], n_total_prompts: int, status: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "method": method,
        "status": status,
        "n_prompts": n_total_prompts,
        "n_selected": len(selected),
    }
    for key in SCORE_KEYS:
        values = [finite_float(item.get(key)) for item in selected]
        values = [value for value in values if value is not None]
        summary[key] = sum(values) / len(values) if values else ""
    return summary


def run_selector(
    method: str,
    grouped: dict[str, list[dict[str, Any]]],
    selector: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    selected: list[dict[str, Any]] = []
    failures = 0
    for prompt_id in sorted(grouped):
        try:
            item = dict(selector(grouped[prompt_id]))
            item["selected_by"] = method
            selected.append(item)
        except (KeyError, ValueError):
            failures += 1
    if failures == len(grouped):
        return [], "not_available"
    if failures:
        return selected, f"partial_missing:{failures}"
    return selected, "completed_from_existing_scores"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selection baselines on an existing candidate score JSONL.")
    parser.add_argument("--scores", required=True, type=Path, help="Candidate score JSONL.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for baseline outputs.")
    parser.add_argument("--random-seeds", default="0,1,2,3,4", help="Comma-separated random seeds.")
    args = parser.parse_args()

    rows = load_rows(args.scores)
    grouped = group_by_prompt(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selectors: list[tuple[str, Callable[[list[dict[str, Any]]], dict[str, Any]]]] = [
        ("fixed_seed0", lambda candidates: fixed_seed_select.select(candidates, seed_value=0)),
        ("clipscore_select", clipscore_select.select),
        ("image_reward_select", image_reward_select.select),
        ("pickscore_select", pickscore_select.select),
        ("hpsv2_select", hpsv2_select.select),
        ("vlm_direct_select", vlm_direct_select.select),
        ("csr_select", csr_select.select),
    ]

    summaries: list[dict[str, Any]] = []
    for method, selector in selectors:
        selected, status = run_selector(method, grouped, selector)
        if selected:
            with (args.out_dir / f"{method}_selected.jsonl").open("w", encoding="utf-8") as handle:
                for item in selected:
                    handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        summaries.append(summarize(method, selected, len(grouped), status))

    random_seeds = [int(seed.strip()) for seed in args.random_seeds.split(",") if seed.strip()]
    random_runs: list[dict[str, Any]] = []
    for seed in random_seeds:
        selected, status = run_selector(
            f"random_select_seed{seed}",
            grouped,
            lambda candidates, seed=seed: random_select.select(candidates, seed=seed),
        )
        random_runs.append(summarize(f"random_select_seed{seed}", selected, len(grouped), status))

    if random_runs:
        random_summary: dict[str, Any] = {
            "method": "random_select_mean",
            "status": "completed_from_existing_scores",
            "n_prompts": len(grouped),
            "n_selected": len(grouped),
        }
        for key in SCORE_KEYS:
            values = [finite_float(row.get(key)) for row in random_runs]
            values = [value for value in values if value is not None]
            random_summary[key] = sum(values) / len(values) if values else ""
            if values:
                mean = float(random_summary[key])
                random_summary[f"{key}_std"] = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        summaries.insert(1, random_summary)

    with (args.out_dir / "baseline_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    with (args.out_dir / "baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
