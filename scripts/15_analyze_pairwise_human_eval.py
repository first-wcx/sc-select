from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    key = {row["task_id"]: row for row in read_csv(args.key)}
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    rows = []
    for answer in read_csv(args.answers):
        task_id = answer["task_id"]
        if task_id not in key:
            continue
        choice = answer.get("choice", "").strip().lower()
        if answer.get("tie", "").strip().lower() in {"1", "true", "yes", "y"} or choice == "tie":
            winner = "tie"
        elif choice in {"left", "right"}:
            winner = key[task_id][f"{choice}_method"]
        else:
            winner = "missing"
        baseline = key[task_id]["baseline"]
        dataset = key[task_id]["dataset"]
        counts[(dataset, baseline)][winner] += 1
        rows.append({
            "task_id": task_id,
            "dataset": dataset,
            "baseline": baseline,
            "winner": winner,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "baseline", "csr_wins", "baseline_wins", "ties", "missing", "csr_win_rate_excl_ties"])
        for (dataset, baseline), counter in sorted(counts.items()):
            csr_wins = counter["csr_select"]
            base_wins = counter[baseline]
            denom = csr_wins + base_wins
            writer.writerow([
                dataset,
                baseline,
                csr_wins,
                base_wins,
                counter["tie"],
                counter["missing"],
                "" if denom == 0 else csr_wins / denom,
            ])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
