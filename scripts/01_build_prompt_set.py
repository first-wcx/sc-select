from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCENES = [
    "in a quiet library",
    "on a city rooftop",
    "inside a small bakery",
    "at a rainy bus stop",
    "in a bright classroom",
    "near a mountain lake",
    "inside a science museum",
    "on a wooden stage",
    "in a modern kitchen",
    "beside a train platform",
]

OBJECTS = [
    "robot", "cat", "dog", "bicycle", "chair", "lamp", "backpack", "teapot", "camera", "kite",
    "car", "umbrella", "book", "vase", "clock", "guitar", "helmet", "basket", "suitcase", "poster",
]

COLORS = ["red", "blue", "yellow", "green", "purple", "white", "black", "orange", "silver", "pink"]
MATERIALS = ["wooden", "metal", "glass", "ceramic", "fabric"]
RELATIONS = ["left_of", "right_of", "above", "below", "in_front_of", "behind", "on", "under", "holding", "next_to"]
TEXTS = ["OPEN", "MARS", "CITY", "AI LAB", "SUN", "MOON", "NOVA", "CAFE", "2026", "GO"]


def obj(i: int) -> str:
    return OBJECTS[i % len(OBJECTS)]


def color(i: int) -> str:
    return COLORS[i % len(COLORS)]


def scene(i: int) -> str:
    return SCENES[i % len(SCENES)]


def relation_phrase(rel: str) -> str:
    return {
        "left_of": "to the left of",
        "right_of": "to the right of",
        "above": "above",
        "below": "below",
        "in_front_of": "in front of",
        "behind": "behind",
        "on": "on",
        "under": "under",
        "holding": "holding",
        "next_to": "next to",
    }[rel]


def base_contract(prompt_id: str, category: str, prompt: str) -> dict:
    return {
        "prompt_id": prompt_id,
        "category": category,
        "prompt": prompt,
        "objects": [],
        "attributes": [],
        "relations": [],
        "counts": [],
        "texts": [],
    }


def build_object_existence() -> list[dict]:
    rows = []
    for i in range(50):
        a, b, c = obj(i), obj(i + 3), obj(i + 7)
        prompt_id = f"complex_obj_{i + 1:04d}"
        prompt = f"A scene {scene(i)} containing a {a}, a {b}, and a {c}, all clearly visible."
        contract = base_contract(prompt_id, "Object Existence", prompt)
        contract["objects"] = [{"name": a}, {"name": b}, {"name": c}]
        rows.append(contract)
    return rows


def build_attributes() -> list[dict]:
    rows = []
    for i in range(50):
        a, b = obj(i), obj(i + 5)
        ca, cb = color(i), color(i + 4)
        material = MATERIALS[i % len(MATERIALS)]
        prompt_id = f"complex_attr_{i + 1:04d}"
        prompt = f"A {ca} {a} beside a {cb} {material} {b} {scene(i)}."
        contract = base_contract(prompt_id, "Attribute Binding", prompt)
        contract["objects"] = [{"name": a}, {"name": b}]
        contract["attributes"] = [
            {"object": a, "attribute_type": "color", "value": ca},
            {"object": b, "attribute_type": "color", "value": cb},
            {"object": b, "attribute_type": "material", "value": material},
        ]
        rows.append(contract)
    return rows


def build_relations() -> list[dict]:
    rows = []
    for i in range(50):
        a, b = obj(i), obj(i + 2)
        rel = RELATIONS[i % len(RELATIONS)]
        prompt_id = f"complex_rel_{i + 1:04d}"
        prompt = f"A {a} is {relation_phrase(rel)} a {b} {scene(i)}."
        contract = base_contract(prompt_id, "Spatial Relation", prompt)
        contract["objects"] = [{"name": a}, {"name": b}]
        contract["relations"] = [{"subject": a, "relation": rel, "object": b}]
        rows.append(contract)
    return rows


def build_counts() -> list[dict]:
    rows = []
    for i in range(50):
        a, b = obj(i), obj(i + 9)
        n = (i % 5) + 1
        prompt_id = f"complex_count_{i + 1:04d}"
        prompt = f"Exactly {n} {a}s are arranged around one {b} {scene(i)}."
        contract = base_contract(prompt_id, "Counting", prompt)
        contract["objects"] = [{"name": a}, {"name": b}]
        contract["counts"] = [{"object": a, "number": n, "operator": "eq"}, {"object": b, "number": 1, "operator": "eq"}]
        rows.append(contract)
    return rows


def build_texts() -> list[dict]:
    rows = []
    for i in range(50):
        a = obj(i)
        text = TEXTS[i % len(TEXTS)]
        prompt_id = f"complex_text_{i + 1:04d}"
        prompt = f"A sign with the readable text '{text}' is placed next to a {a} {scene(i)}."
        contract = base_contract(prompt_id, "Text Rendering", prompt)
        contract["objects"] = [{"name": "sign"}, {"name": a}]
        contract["relations"] = [{"subject": "sign", "relation": "next_to", "object": a}]
        contract["texts"] = [{"content": text, "case_sensitive": False}]
        rows.append(contract)
    return rows


def build_complex() -> list[dict]:
    rows = []
    for i in range(50):
        a, b, c = obj(i), obj(i + 4), obj(i + 8)
        ca, cb = color(i), color(i + 5)
        rel = RELATIONS[i % len(RELATIONS)]
        text = TEXTS[(i + 3) % len(TEXTS)]
        n = (i % 3) + 2
        prompt_id = f"complex_combo_{i + 1:04d}"
        prompt = (
            f"A {ca} {a} is {relation_phrase(rel)} a {cb} {b}, with exactly {n} {c}s nearby, "
            f"and a small label reads '{text}' {scene(i)}."
        )
        contract = base_contract(prompt_id, "Complex Composition", prompt)
        contract["objects"] = [{"name": a}, {"name": b}, {"name": c}, {"name": "label"}]
        contract["attributes"] = [
            {"object": a, "attribute_type": "color", "value": ca},
            {"object": b, "attribute_type": "color", "value": cb},
        ]
        contract["relations"] = [{"subject": a, "relation": rel, "object": b}]
        contract["counts"] = [{"object": c, "number": n, "operator": "eq"}]
        contract["texts"] = [{"content": text, "case_sensitive": False}]
        rows.append(contract)
    return rows


def main() -> None:
    contracts = (
        build_object_existence()
        + build_attributes()
        + build_relations()
        + build_counts()
        + build_texts()
        + build_complex()
    )
    prompt_path = ROOT / "data" / "prompts" / "scselect_complex_300.jsonl"
    contract_path = ROOT / "data" / "contracts" / "scselect_complex_300_contracts.jsonl"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    with prompt_path.open("w", encoding="utf-8") as prompt_handle, contract_path.open("w", encoding="utf-8") as contract_handle:
        for contract in contracts:
            prompt_row = {
                "prompt_id": contract["prompt_id"],
                "category": contract["category"],
                "prompt": contract["prompt"],
                "contract": contract,
                "source": "semi-automated template expansion",
                "checked": False,
                "review_status": "needs_manual_review",
            }
            prompt_handle.write(json.dumps(prompt_row, ensure_ascii=False) + "\n")
            contract_handle.write(json.dumps(contract, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

