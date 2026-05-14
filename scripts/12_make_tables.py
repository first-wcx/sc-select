from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "server_audits"
OUT = ROOT / "outputs" / "tables"
PAPER = ROOT / "paper" / "tables"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_md_table(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 3:
        return []
    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [re.sub(r"[*`]", "", cell.strip()) for cell in line.strip("|").split("|")]
        rows.append(dict(zip(headers, cells)))
    return rows


def pct_to_float(value: str) -> float:
    return float(value.replace("%", "").replace("+", "").replace(" pp", "").strip())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    # Self-built 50 prompt result from server_004.
    self_rows = parse_md_table(AUDITS / "server_004" / "paper_assets" / "tables" / "main_result.md")
    write_csv(OUT / "self_built_50_csr_results.csv", self_rows)
    write_csv(PAPER / "self_built_50_csr_results.csv", self_rows)

    # GenEval CSR and official evaluator results.
    geneval_csr = parse_md_table(AUDITS / "server_004" / "paper_assets" / "tables" / "geneval_csr_result.md")
    write_csv(OUT / "geneval_csr_results.csv", geneval_csr)
    write_csv(PAPER / "geneval_csr_results.csv", geneval_csr)

    geneval_official = parse_md_table(
        AUDITS / "server_004" / "paper_assets" / "tables" / "geneval_official_result.md"
    )
    write_csv(OUT / "geneval_official_results.csv", geneval_official)
    write_csv(PAPER / "geneval_official_results.csv", geneval_official)

    vqa_rows = parse_md_table(AUDITS / "server_004" / "paper_assets" / "tables" / "vqa_select_result.md")
    write_csv(OUT / "vqa_baseline_results.csv", vqa_rows)
    write_csv(PAPER / "vqa_baseline_results.csv", vqa_rows)

    # Cross-generator table: only rows backed by audited artifacts are included.
    flux_table = parse_md_table(AUDITS / "server_001" / "paper_assets" / "tables" / "multimodel_flux_result.md")
    cross_rows = []
    for row in flux_table:
        if row.get("Generator") == "FLUX.1-schnell":
            cross_rows.append(
                {
                    "generator": "FLUX.1-schnell",
                    "dataset": "GenEval",
                    "n_prompts": 553,
                    "n_candidates": 4,
                    "method": "SC-Select Best-of-4",
                    "csr_all": row["SC-Select Best-of-4 CSR-All"],
                    "gain_vs_all_candidates_mean": row["Gain"],
                    "source": "server_001 audited FLUX GenEval output",
                }
            )
        elif row.get("Generator") == "SDXL":
            cross_rows.append(
                {
                    "generator": "SDXL",
                    "dataset": "GenEval",
                    "n_prompts": 553,
                    "n_candidates": 4,
                    "method": "SC-Select Best-of-4",
                    "csr_all": row["SC-Select Best-of-4 CSR-All"],
                    "gain_vs_all_candidates_mean": row["Gain"],
                    "source": "server_004 audited SDXL GenEval table",
                }
            )
    write_csv(OUT / "cross_generator_results.csv", cross_rows)
    write_csv(PAPER / "cross_generator_results.csv", cross_rows)

    provenance = {
        "self_built_50_csr_results.csv": "server_004/paper_assets/tables/main_result.md",
        "geneval_csr_results.csv": "server_004/paper_assets/tables/geneval_csr_result.md",
        "geneval_official_results.csv": "server_004/paper_assets/tables/geneval_official_result.md",
        "vqa_baseline_results.csv": "server_004/paper_assets/tables/vqa_select_result.md",
        "cross_generator_results.csv": "server_001 and server_004 audited tables",
    }
    (OUT / "table_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

