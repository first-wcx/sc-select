from __future__ import annotations

from typing import Any


def select(candidates: list[dict[str, Any]], seed_value: int = 0) -> dict[str, Any]:
    if not candidates:
        raise ValueError("fixed_seed_select requires at least one candidate")
    for candidate in candidates:
        if candidate.get("seed") == seed_value:
            return candidate
    return sorted(candidates, key=lambda item: str(item.get("image_id", "")))[0]

