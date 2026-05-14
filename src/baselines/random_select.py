from __future__ import annotations

import random
from typing import Any


def select(candidates: list[dict[str, Any]], seed: int | None = None) -> dict[str, Any]:
    if not candidates:
        raise ValueError("random_select requires at least one candidate")
    rng = random.Random(seed)
    return rng.choice(candidates)

