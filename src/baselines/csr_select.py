from __future__ import annotations

from typing import Any


def select(scored_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored_candidates:
        raise ValueError("csr_select requires at least one scored candidate")
    return max(scored_candidates, key=lambda item: float(item.get("csr_all") or -1.0))

