from __future__ import annotations

from typing import Any

from .score_select import select as select_by_score


def select(scored_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return select_by_score(scored_candidates, "clipscore")
    except KeyError:
        return select_by_score(scored_candidates, "clip_score")
