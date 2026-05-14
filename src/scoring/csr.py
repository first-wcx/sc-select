from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


OBJECT_ALIASES = {
    "sofa": {"couch"},
    "couch": {"sofa"},
    "cup": {"mug"},
    "mug": {"cup"},
    "bike": {"bicycle"},
    "bicycle": {"bike"},
}

RELATION_ALIASES = {
    "beside": "next_to",
    "next to": "next_to",
    "near": "next_to",
    "left of": "left_of",
    "right of": "right_of",
    "in front of": "in_front_of",
    "front of": "in_front_of",
    "on top of": "on",
    "beneath": "under",
    "below": "under",
    "inside": "in",
}


@dataclass
class PartialScore:
    value: float | None
    satisfied: int
    total: int
    violations: list[dict[str, str]]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_object_name(name: Any) -> str:
    text = normalize_text(name)
    if text.endswith("s") and len(text) > 3:
        return text[:-1]
    return text


def object_names_match(expected: str, actual: str, aliases: list[str] | None = None) -> bool:
    expected_norm = normalize_object_name(expected)
    actual_norm = normalize_object_name(actual)
    if expected_norm == actual_norm:
        return True
    alias_set = {normalize_object_name(item) for item in aliases or []}
    alias_set |= OBJECT_ALIASES.get(expected_norm, set())
    return actual_norm in alias_set


def normalize_relation(relation: Any) -> str:
    text = normalize_text(relation).replace("-", "_")
    text = RELATION_ALIASES.get(text, text)
    return text.replace(" ", "_")


def text_match(expected: str, candidates: list[str], case_sensitive: bool = False) -> bool:
    def clean(value: str) -> str:
        if not case_sensitive:
            value = value.lower()
        return re.sub(r"[^a-z0-9]+", "", value)

    expected_clean = clean(expected)
    return any(expected_clean and expected_clean in clean(candidate) for candidate in candidates)


def _visible_objects(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return list(graph.get("visible_objects") or [])


def _find_object(contract_obj: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any] | None:
    expected = contract_obj.get("name") or contract_obj.get("object")
    aliases = contract_obj.get("aliases") or []
    for item in _visible_objects(graph):
        if object_names_match(expected, item.get("name", ""), aliases):
            return item
    return None


def score_objects(contract: dict[str, Any], graph: dict[str, Any]) -> PartialScore:
    objects = contract.get("objects") or []
    violations: list[dict[str, str]] = []
    if not objects:
        return PartialScore(None, 0, 0, violations)
    satisfied = 0
    for obj in objects:
        if _find_object(obj, graph):
            satisfied += 1
        else:
            violations.append({
                "type": "object",
                "contract": str(obj),
                "reason": f"Expected object '{obj.get('name')}' was not found",
            })
    return PartialScore(satisfied / len(objects), satisfied, len(objects), violations)


def score_attributes(contract: dict[str, Any], graph: dict[str, Any]) -> PartialScore:
    attrs = contract.get("attributes") or []
    violations: list[dict[str, str]] = []
    if not attrs:
        return PartialScore(None, 0, 0, violations)
    satisfied = 0
    for attr in attrs:
        obj = _find_object({"name": attr.get("object")}, graph)
        expected_type = normalize_text(attr.get("attribute_type"))
        expected_value = normalize_text(attr.get("value"))
        actual_value = normalize_text((obj or {}).get("attributes", {}).get(expected_type))
        if obj and expected_value and expected_value == actual_value:
            satisfied += 1
        else:
            violations.append({
                "type": "attribute",
                "contract": str(attr),
                "reason": f"{attr.get('object')} should have {expected_type}={expected_value}, parsed as {actual_value or 'missing'}",
            })
    return PartialScore(satisfied / len(attrs), satisfied, len(attrs), violations)


def score_relations(contract: dict[str, Any], graph: dict[str, Any]) -> PartialScore:
    required = contract.get("relations") or []
    parsed = graph.get("relations") or []
    violations: list[dict[str, str]] = []
    if not required:
        return PartialScore(None, 0, 0, violations)
    satisfied = 0
    for rel in required:
        expected_subject = rel.get("subject", "")
        expected_object = rel.get("object", "")
        expected_relation = normalize_relation(rel.get("relation"))
        matched = False
        for item in parsed:
            if (
                object_names_match(expected_subject, item.get("subject", ""))
                and object_names_match(expected_object, item.get("object", ""))
                and normalize_relation(item.get("relation")) == expected_relation
            ):
                matched = True
                break
        if matched:
            satisfied += 1
        else:
            violations.append({
                "type": "relation",
                "contract": str(rel),
                "reason": "Expected relation triple was not found",
            })
    return PartialScore(satisfied / len(required), satisfied, len(required), violations)


def score_counts(contract: dict[str, Any], graph: dict[str, Any]) -> PartialScore:
    counts = contract.get("counts") or []
    violations: list[dict[str, str]] = []
    if not counts:
        return PartialScore(None, 0, 0, violations)
    satisfied = 0
    for count in counts:
        obj = _find_object({"name": count.get("object")}, graph)
        actual = (obj or {}).get("count")
        expected = count.get("number")
        operator = count.get("operator", "eq")
        ok = False
        if isinstance(actual, int) and isinstance(expected, int):
            if operator == "eq":
                ok = actual == expected
            elif operator == "gte":
                ok = actual >= expected
            elif operator == "lte":
                ok = actual <= expected
        if ok:
            satisfied += 1
        else:
            violations.append({
                "type": "count",
                "contract": str(count),
                "reason": f"Expected {operator} {expected}, parsed as {actual if actual is not None else 'missing'}",
            })
    return PartialScore(satisfied / len(counts), satisfied, len(counts), violations)


def score_texts(contract: dict[str, Any], graph: dict[str, Any]) -> PartialScore:
    texts = contract.get("texts") or []
    candidates = list(graph.get("ocr_texts") or [])
    violations: list[dict[str, str]] = []
    if not texts:
        return PartialScore(None, 0, 0, violations)
    satisfied = 0
    for text in texts:
        expected = text.get("content", "")
        if text_match(expected, candidates, bool(text.get("case_sensitive", False))):
            satisfied += 1
        else:
            violations.append({
                "type": "text",
                "contract": str(text),
                "reason": f"Expected OCR text '{expected}' was not found",
            })
    return PartialScore(satisfied / len(texts), satisfied, len(texts), violations)


def score_all(contract: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    parts = {
        "csr_obj": score_objects(contract, graph),
        "csr_attr": score_attributes(contract, graph),
        "csr_rel": score_relations(contract, graph),
        "csr_count": score_counts(contract, graph),
        "csr_text": score_texts(contract, graph),
    }
    valid_values = [part.value for part in parts.values() if part.value is not None]
    violations: list[dict[str, str]] = []
    for part in parts.values():
        violations.extend(part.violations)
    return {
        "prompt_id": contract.get("prompt_id") or graph.get("prompt_id"),
        "image_id": graph.get("image_id"),
        "csr_obj": parts["csr_obj"].value,
        "csr_attr": parts["csr_attr"].value,
        "csr_rel": parts["csr_rel"].value,
        "csr_count": parts["csr_count"].value,
        "csr_text": parts["csr_text"].value,
        "csr_all": sum(valid_values) / len(valid_values) if valid_values else None,
        "violations": violations,
    }

