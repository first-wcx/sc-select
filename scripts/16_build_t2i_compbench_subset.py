from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

COLORS = {"red", "blue", "green", "brown", "black", "white", "yellow", "orange", "purple", "pink", "gray", "grey"}

RELATION_ALIASES = [
    ("on the top of", "above"),
    ("on top of", "above"),
    ("on the bottom of", "below"),
    ("on the left of", "left_of"),
    ("on the right of", "right_of"),
    ("next to", "beside"),
    ("on side of", "beside"),
    ("inside", "inside"),
    ("resting on", "on"),
    ("on", "on"),
]


def singular(noun: str) -> str:
    noun = noun.strip().lower()
    if noun.endswith("ies"):
        return noun[:-3] + "y"
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def parse_color(prompt: str, prompt_id: str) -> dict | None:
    match = re.match(r"^a ([a-z]+) ([a-z ]+?) and a ([a-z]+) ([a-z ]+)$", prompt.lower())
    if not match:
        return None
    c1, o1, c2, o2 = match.groups()
    if c1 not in COLORS or c2 not in COLORS:
        return None
    o1 = singular(o1)
    o2 = singular(o2)
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "objects": [{"name": o1}, {"name": o2}],
        "attributes": [
            {"object": o1, "attribute_type": "color", "value": c1},
            {"object": o2, "attribute_type": "color", "value": c2},
        ],
        "relations": [],
        "counts": [],
        "texts": [],
        "checked": False,
        "source": "T2I-CompBench/color_val",
    }


def parse_spatial(prompt: str, prompt_id: str) -> dict | None:
    lower = prompt.lower().strip().rstrip(".")
    if lower.startswith("the "):
        lower = lower[4:]
    if lower.startswith("a "):
        lower = lower[2:]
    if lower.startswith("an "):
        lower = lower[3:]
    for phrase, rel in RELATION_ALIASES:
        marker = f" {phrase} "
        if marker not in lower:
            continue
        left, right = lower.split(marker, 1)
        right = re.sub(r"^(a|an|the) ", "", right)
        subj = singular(left)
        obj = singular(right)
        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "objects": [{"name": subj}, {"name": obj}],
            "attributes": [],
            "relations": [{"subject": subj, "relation": rel, "object": obj}],
            "counts": [],
            "texts": [],
            "checked": False,
            "source": "T2I-CompBench/spatial_val",
        }
    return None


def parse_numeracy(prompt: str, prompt_id: str) -> dict | None:
    match = re.match(r"^([a-z]+) ([a-z ]+)$", prompt.lower().strip())
    if not match:
        return None
    number_word, obj = match.groups()
    if number_word not in NUMBERS:
        return None
    obj = singular(obj)
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "objects": [{"name": obj}],
        "attributes": [],
        "relations": [],
        "counts": [{"object": obj, "number": NUMBERS[number_word]}],
        "texts": [],
        "checked": False,
        "source": "T2I-CompBench/numeracy_val",
    }


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-prompts", type=Path, required=True)
    parser.add_argument("--out-contracts", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=20)
    args = parser.parse_args()

    rows: list[dict] = []
    specs = [
        ("color", "color_val.txt", parse_color),
        ("spatial", "spatial_val.txt", parse_spatial),
        ("numeracy", "numeracy_val.txt", parse_numeracy),
    ]
    for category, file_name, parse_fn in specs:
        count = 0
        for prompt in read_lines(args.raw_dir / file_name):
            prompt_id = f"t2icomp_{category}_{count + 1:04d}"
            row = parse_fn(prompt, prompt_id)
            if row is None:
                continue
            row["category"] = category
            rows.append(row)
            count += 1
            if count >= args.per_category:
                break

    args.out_prompts.parent.mkdir(parents=True, exist_ok=True)
    args.out_contracts.parent.mkdir(parents=True, exist_ok=True)
    with args.out_prompts.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"prompt_id": row["prompt_id"], "prompt": row["prompt"], "category": row["category"]}, ensure_ascii=False) + "\n")
    with args.out_contracts.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
