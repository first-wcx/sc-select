from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.statistics import paired_bootstrap_ci, sign_test_p_value


def read_scores(path: Path, method_a: str, method_b: str, metric: str) -> tuple[list[float], list[float]]:
    rows_by_prompt: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt_id = row["prompt_id"]
            method = row["method"]
            if method not in {method_a, method_b}:
                continue
            rows_by_prompt.setdefault(prompt_id, {})[method] = float(row[metric])
    a_values: list[float] = []
    b_values: list[float] = []
    for values in rows_by_prompt.values():
        if method_a in values and method_b in values:
            a_values.append(values[method_a])
            b_values.append(values[method_b])
    return a_values, b_values


def read_jsonl_scores(path: Path, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            value = row.get(metric)
            if value is not None:
                out[str(row["prompt_id"])] = float(value)
    return out


def read_selected_scores(path_a: Path, path_b: Path, metric: str) -> tuple[list[float], list[float]]:
    scores_a = read_jsonl_scores(path_a, metric)
    scores_b = read_jsonl_scores(path_b, metric)
    prompt_ids = sorted(set(scores_a) & set(scores_b))
    return [scores_a[prompt_id] for prompt_id in prompt_ids], [scores_b[prompt_id] for prompt_id in prompt_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--selected-a", type=Path)
    parser.add_argument("--selected-b", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metric", default="csr_all")
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()

    if args.selected_a and args.selected_b:
        a_values, b_values = read_selected_scores(args.selected_a, args.selected_b, args.metric)
    elif args.input:
        a_values, b_values = read_scores(args.input, args.method_a, args.method_b, args.metric)
    else:
        raise SystemExit("provide either --input or both --selected-a and --selected-b")
    result = paired_bootstrap_ci(a_values, b_values, n_resamples=args.resamples)
    result["p_value_sign_test"] = sign_test_p_value(a_values, b_values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["dataset", "metric", "method_a", "method_b", "mean_diff", "ci_low", "ci_high", "p_value_sign_test"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "dataset": args.dataset,
            "metric": args.metric,
            "method_a": args.method_a,
            "method_b": args.method_b,
            **result,
        })


if __name__ == "__main__":
    main()
