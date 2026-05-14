#!/usr/bin/env python3
"""Generic CSR scoring for SC-Select perception graphs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm


REL_ALIASES = {
    "left_of": ["left", "left_of", "to_the_left_of", "beside", "next_to"],
    "right_of": ["right", "right_of", "to_the_right_of", "beside", "next_to"],
    "above": ["above", "over", "on_top_of", "on"],
    "below": ["below", "under", "beneath", "underneath"],
    "in_front_of": ["in_front_of", "in_front", "before", "ahead_of"],
    "behind": ["behind", "back_of", "at_the_back"],
    "holding": ["holding", "carrying", "grasping", "wielding"],
    "beside": ["beside", "next_to", "adjacent_to", "near", "close_to"],
    "inside": ["inside", "in", "within", "contained_in"],
    "on": ["on", "on_top_of", "resting_on", "placed_on"],
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_objects(contract: dict, graph: dict) -> float | None:
    contract_objects = {o["name"].lower() for o in contract.get("objects", [])}
    if not contract_objects:
        return None
    graph_objects = set()
    for obj in graph.get("visible_objects", []):
        if isinstance(obj, dict):
            graph_objects.add(obj.get("name", "").lower())
        elif isinstance(obj, str):
            graph_objects.add(obj.lower())
    return len(contract_objects & graph_objects) / len(contract_objects)


def score_attributes(contract: dict, graph: dict) -> float | None:
    contract_attrs = contract.get("attributes", [])
    if not contract_attrs:
        return None
    graph_attrs = []
    for obj in graph.get("visible_objects", []):
        if not isinstance(obj, dict):
            continue
        obj_name = obj.get("name", "").lower()
        attrs = obj.get("attributes", {})
        if isinstance(attrs, dict):
            for key, value in attrs.items():
                graph_attrs.append((obj_name, key.lower(), str(value).lower()))
    matched = 0
    for attr in contract_attrs:
        obj = attr.get("object", "").lower()
        attr_type = attr.get("attribute_type", "").lower()
        value = attr.get("value", "").lower()
        for graph_obj, graph_type, graph_value in graph_attrs:
            if graph_obj == obj and (attr_type in graph_type or graph_type in attr_type):
                if value in graph_value or graph_value in value:
                    matched += 1
                    break
    return matched / len(contract_attrs)


def score_relations(contract: dict, graph: dict) -> float | None:
    contract_rels = contract.get("relations", [])
    if not contract_rels:
        return None
    graph_rels = graph.get("relations", [])
    if not graph_rels:
        return 0.0
    matched = 0
    for rel_row in contract_rels:
        subj = rel_row.get("subject", "").lower()
        rel = rel_row.get("relation", "").lower()
        obj = rel_row.get("object", "").lower()
        aliases = REL_ALIASES.get(rel, [rel])
        for graph_rel in graph_rels:
            if not isinstance(graph_rel, dict):
                continue
            graph_subj = graph_rel.get("subject", "").lower()
            graph_rel_name = graph_rel.get("relation", "").lower()
            graph_obj = graph_rel.get("object", "").lower()
            if graph_subj == subj and graph_obj == obj:
                if any(alias in graph_rel_name or graph_rel_name in alias for alias in aliases):
                    matched += 1
                    break
    return matched / len(contract_rels)


def score_counts(contract: dict, graph: dict) -> float | None:
    contract_counts = contract.get("counts", [])
    if not contract_counts:
        return None
    graph_counts = graph.get("count", [])
    if not graph_counts:
        graph_counts = []
        for obj in graph.get("visible_objects", []):
            if isinstance(obj, dict):
                graph_counts.append({"object": obj.get("name", ""), "number": obj.get("count", 1)})
    matched = 0
    for count_row in contract_counts:
        obj = count_row.get("object", "").lower()
        expected = count_row.get("number", 0)
        for graph_count in graph_counts:
            graph_obj = graph_count.get("object", "").lower() if isinstance(graph_count, dict) else ""
            graph_num = graph_count.get("number", 0) if isinstance(graph_count, dict) else 0
            if graph_obj == obj and graph_num == expected:
                matched += 1
                break
    return matched / len(contract_counts)


def score_texts(contract: dict, graph: dict) -> float | None:
    contract_texts = contract.get("texts", [])
    if not contract_texts:
        return None
    ocr_texts = graph.get("ocr_text", [])
    if isinstance(ocr_texts, str):
        ocr_texts = [ocr_texts]
    ocr_lower = " ".join(str(text).lower() for text in ocr_texts)
    matched = 0
    for text_row in contract_texts:
        content = text_row.get("content", "").lower()
        if content in ocr_lower:
            matched += 1
    return matched / len(contract_texts)


def score_all(contract: dict, graph: dict) -> dict[str, float | None]:
    scores = {
        "csr_obj": score_objects(contract, graph),
        "csr_attr": score_attributes(contract, graph),
        "csr_rel": score_relations(contract, graph),
        "csr_count": score_counts(contract, graph),
        "csr_text": score_texts(contract, graph),
    }
    valid = [score for score in scores.values() if score is not None]
    scores["csr_all"] = sum(valid) / len(valid) if valid else 0.0
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-file", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--perception-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    contracts = {row["prompt_id"]: row for row in load_jsonl(Path(args.contract_file))}
    prompts = {row["prompt_id"]: row["prompt"] for row in load_jsonl(Path(args.prompt_file))}
    perception_files = sorted(Path(args.perception_dir).glob("*.json"))
    image_dir = Path(args.image_dir)
    print(f"Perception graphs: {len(perception_files)}", flush=True)
    print(f"Contracts: {len(contracts)}", flush=True)

    results = []
    for perception_file in tqdm(perception_files, desc="CSR scoring"):
        with perception_file.open("r", encoding="utf-8") as handle:
            graph = json.load(handle)
        pid = graph.get("prompt_id", perception_file.stem.rsplit("_", 1)[0])
        image_id = graph.get("image_id", perception_file.stem.rsplit("_", 1)[-1])
        contract = contracts.get(pid)
        if not contract:
            continue
        scores = score_all(contract, graph)
        scores["prompt_id"] = pid
        scores["image_id"] = image_id
        scores["prompt"] = prompts.get(pid, "")
        scores["image_path"] = str(image_dir / pid / f"{image_id}.png")
        results.append(scores)

    out_jsonl = out_dir / "csr_results.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if results:
        import pandas as pd

        df = pd.DataFrame(results)
        df.to_csv(out_dir / "candidate_scores.csv", index=False)
        for col in ["csr_obj", "csr_attr", "csr_rel", "csr_count", "csr_text", "csr_all"]:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"{col}: mean={vals.mean():.4f}, std={vals.std():.4f}", flush=True)

    print(f"CSR scoring done: {len(results)} results", flush=True)
    print(f"Saved to: {out_jsonl}", flush=True)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
