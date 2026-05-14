from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


def pairwise_agreement(labels_by_item: dict[str, list[str]]) -> float:
    agreements = 0
    pairs = 0
    for labels in labels_by_item.values():
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs += 1
                agreements += int(labels[i] == labels[j])
    return agreements / pairs if pairs else 0.0


def majority_vote(labels: Iterable[str]) -> str | None:
    counts = Counter(labels)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def win_rates(records: list[dict[str, str]], method_by_candidate: dict[str, str]) -> dict[str, float]:
    wins: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    by_prompt: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_prompt[record["prompt_id"]].append(record["best_candidate_id"])
    for prompt_id, labels in by_prompt.items():
        winner = majority_vote(labels)
        if winner is None:
            continue
        winner_method = method_by_candidate.get(winner)
        methods = {method_by_candidate.get(label) for label in labels if method_by_candidate.get(label)}
        for method in methods:
            totals[method] += 1
        if winner_method:
            wins[winner_method] += 1
    return {method: wins[method] / total for method, total in totals.items() if total}

