from __future__ import annotations

from typing import Any

from .score_select import select as select_by_score


def select(scored_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return select_by_score(scored_candidates, "hpsv2")

