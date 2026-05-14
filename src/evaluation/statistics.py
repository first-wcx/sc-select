from __future__ import annotations

import math
import random
from typing import Iterable


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def paired_differences(method_a: list[float], method_b: list[float]) -> list[float]:
    if len(method_a) != len(method_b):
        raise ValueError("paired inputs must have equal length")
    return [a - b for a, b in zip(method_a, method_b)]


def paired_bootstrap_ci(
    method_a: list[float],
    method_b: list[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 13,
) -> dict[str, float]:
    diffs = paired_differences(method_a, method_b)
    if not diffs:
        raise ValueError("bootstrap requires at least one paired observation")
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot_means.append(mean(sample))
    boot_means.sort()
    alpha = 1.0 - confidence
    low_idx = max(0, math.floor((alpha / 2.0) * len(boot_means)))
    high_idx = min(len(boot_means) - 1, math.ceil((1.0 - alpha / 2.0) * len(boot_means)) - 1)
    return {
        "mean_diff": mean(diffs),
        "ci_low": boot_means[low_idx],
        "ci_high": boot_means[high_idx],
    }


def sign_test_p_value(method_a: list[float], method_b: list[float]) -> float:
    diffs = [diff for diff in paired_differences(method_a, method_b) if diff != 0]
    n = len(diffs)
    if n == 0:
        return 1.0
    positives = sum(1 for diff in diffs if diff > 0)
    k = min(positives, n - positives)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * prob)

