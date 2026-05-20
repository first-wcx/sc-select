#!/usr/bin/env python3
"""CSR scoring and baseline comparison for 300-prompt dataset.

Run AFTER Qwen parsing completes.
CPU-only, no GPU needed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CONTRACT_FILE = Path("data/scselect_complex_300_contracts.jsonl")
PERCEPTION_DIR = Path("outputs/perception_graphs_300")
PROMPT_FILE = Path("data/scselect_complex_300.jsonl")
OUTPUT_DIR = Path("outputs/csr_scores_300")
LOG_FILE = Path("outputs/logs_300/csr_scoring.log")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

log = open(LOG_FILE, "w", encoding="utf-8")

def logmsg(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def score_objects(contract, graph):
    """Score object presence."""
    contract_objects = {o["name"].lower() for o in contract.get("objects", [])}
    if not contract_objects:
        return None
    graph_objects = set()
    for o in graph.get("visible_objects", []):
        if isinstance(o, dict):
            graph_objects.add(o.get("name", "").lower())
        elif isinstance(o, str):
            graph_objects.add(o.lower())
    found = contract_objects & graph_objects
    return len(found) / len(contract_objects) if contract_objects else None


def score_attributes(contract, graph):
    """Score attribute binding."""
    contract_attrs = contract.get("attributes", [])
    if not contract_attrs:
        return None
    graph_attrs = []
    for o in graph.get("visible_objects", []):
        if isinstance(o, dict):
            obj_name = o.get("name", "").lower()
            attrs = o.get("attributes", {})
            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    graph_attrs.append((obj_name, k.lower(), str(v).lower()))
    matched = 0
    for ca in contract_attrs:
        obj = ca.get("object", "").lower()
        attr_type = ca.get("attribute_type", "").lower()
        value = ca.get("value", "").lower()
        for ga_obj, ga_type, ga_val in graph_attrs:
            if ga_obj == obj and (attr_type in ga_type or ga_type in attr_type):
                if value in ga_val or ga_val in value:
                    matched += 1
                    break
    return matched / len(contract_attrs)


def score_relations(contract, graph):
    """Score spatial/action relations."""
    contract_rels = contract.get("relations", [])
    if not contract_rels:
        return None
    graph_rels = graph.get("relations", [])
    if not graph_rels:
        return 0.0

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

    matched = 0
    for cr in contract_rels:
        subj = cr.get("subject", "").lower()
        rel = cr.get("relation", "").lower()
        obj = cr.get("object", "").lower()
        aliases = REL_ALIASES.get(rel, [rel])
        for gr in graph_rels:
            if isinstance(gr, dict):
                g_subj = gr.get("subject", "").lower()
                g_rel = gr.get("relation", "").lower()
                g_obj = gr.get("object", "").lower()
                if g_subj == subj and g_obj == obj:
                    if any(a in g_rel or g_rel in a for a in aliases):
                        matched += 1
                        break
    return matched / len(contract_rels)


def score_counts(contract, graph):
    """Score counting constraints."""
    contract_counts = contract.get("counts", [])
    if not contract_counts:
        return None
    graph_counts = graph.get("count", [])
    if not graph_counts:
        graph_counts = []
        for o in graph.get("visible_objects", []):
            if isinstance(o, dict):
                graph_counts.append({"object": o.get("name", ""), "number": o.get("count", 1)})

    matched = 0
    for cc in contract_counts:
        obj = cc.get("object", "").lower()
        expected = cc.get("number", 0)
        for gc in graph_counts:
            gc_obj = gc.get("object", "").lower() if isinstance(gc, dict) else ""
            gc_num = gc.get("number", 0) if isinstance(gc, dict) else 0
            if gc_obj == obj and gc_num == expected:
                matched += 1
                break
    return matched / len(contract_counts)


def score_texts(contract, graph):
    """Score text/OCR constraints."""
    contract_texts = contract.get("texts", [])
    if not contract_texts:
        return None
    ocr_texts = graph.get("ocr_text", [])
    if isinstance(ocr_texts, str):
        ocr_texts = [ocr_texts]
    ocr_lower = " ".join(str(t).lower() for t in ocr_texts)

    matched = 0
    for ct in contract_texts:
        content = ct.get("content", "").lower()
        case_sensitive = ct.get("case_sensitive", False)
        if case_sensitive:
            ocr_join = " ".join(str(t) for t in ocr_texts)
            if content in ocr_join:
                matched += 1
        else:
            if content in ocr_lower:
                matched += 1
    return matched / len(contract_texts)


def score_all(contract, graph):
    """Compute all CSR scores."""
    obj = score_objects(contract, graph)
    attr = score_attributes(contract, graph)
    rel = score_relations(contract, graph)
    count = score_counts(contract, graph)
    text = score_texts(contract, graph)

    scores = {
        "csr_obj": obj,
        "csr_attr": attr,
        "csr_rel": rel,
        "csr_count": count,
        "csr_text": text,
    }

    # CSR-All: average of non-None scores
    valid = [v for v in scores.values() if v is not None]
    scores["csr_all"] = sum(valid) / len(valid) if valid else 0.0

    return scores


def main():
    logmsg(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load contracts
    contracts = {}
    with open(CONTRACT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                contracts[c["prompt_id"]] = c

    # Load prompts
    prompts_map = {}
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                prompts_map[p["prompt_id"]] = p["prompt"]

    # Find perception graphs
    pg_files = sorted(PERCEPTION_DIR.glob("*.json"))
    logmsg(f"Perception graphs: {len(pg_files)}")
    logmsg(f"Contracts: {len(contracts)}")

    results = []
    for pg_file in tqdm(pg_files, desc="CSR scoring"):
        with open(pg_file, "r", encoding="utf-8") as f:
            pg = json.load(f)

        pid = pg.get("prompt_id", pg_file.stem.rsplit("_", 1)[0])
        img_id = pg.get("image_id", pg_file.stem.rsplit("_", 1)[-1])
        contract = contracts.get(pid)
        if not contract:
            continue

        scores = score_all(contract, pg)
        scores["prompt_id"] = pid
        scores["image_id"] = img_id
        scores["prompt"] = prompts_map.get(pid, "")
        results.append(scores)

    # Save results
    output_file = OUTPUT_DIR / "csr_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logmsg(f"CSR scoring done: {len(results)} results")
    logmsg(f"Saved to: {output_file}")

    # Summary stats
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        for col in ["csr_obj", "csr_attr", "csr_rel", "csr_count", "csr_text", "csr_all"]:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                logmsg(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    logmsg(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.close()


if __name__ == "__main__":
    from tqdm import tqdm
    main()
