from __future__ import annotations

from typing import Any


def select(scored_candidates: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    if not scored_candidates:
        raise ValueError(f"{score_key} selection requires at least one candidate")
    available = [item for item in scored_candidates if item.get(score_key) is not None]
    if not available:
        raise KeyError(f"score field not available: {score_key}")
    return max(available, key=lambda item: float(item[score_key]))

