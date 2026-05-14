from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs" / "figures"
PAPER = ROOT / "paper" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_float(value: str) -> float:
    return float(value.replace("%", "").replace("+", "").replace(" pp", "").strip())


def bar_svg(title: str, labels: list[str], values: list[float], path: Path, ylabel: str = "CSR-All") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 440
    left, top, bottom = 90, 70, 80
    plot_w, plot_h = width - left - 40, height - top - bottom
    max_v = max(1.0, max(values) * 1.12)
    bar_w = plot_w / max(1, len(values)) * 0.58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<text x="24" y="{top+plot_h/2}" transform="rotate(-90 24 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="14">{ylabel}</text>',
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + (i + 0.5) * plot_w / len(values) - bar_w / 2
        h = value / max_v * plot_h
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[i % len(colors)]}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="13">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top+plot_h+26}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - (tick / max_v * plot_h)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def copy_to_paper(name: str) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / name).write_text((OUT / name).read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    self_rows = read_csv(TABLES / "self_built_50_csr_results.csv")
    self_values = [clean_float(row["CSR-All"]) for row in self_rows if row["Method"] != "Gain"]
    self_labels = [row["Method"].replace(" Best-of-4", "") for row in self_rows if row["Method"] != "Gain"]
    bar_svg("Self-built 50 Prompt CSR-All", self_labels, self_values, OUT / "self_built_50_csr_all.svg")
    copy_to_paper("self_built_50_csr_all.svg")

    geneval_rows = read_csv(TABLES / "geneval_csr_results.csv")
    geneval_values = [clean_float(row["CSR-All"]) for row in geneval_rows if row["Method"] != "Gain"]
    geneval_labels = [row["Method"].replace(" Best-of-4", "") for row in geneval_rows if row["Method"] != "Gain"]
    bar_svg("GenEval CSR-All", geneval_labels, geneval_values, OUT / "geneval_csr_all.svg")
    copy_to_paper("geneval_csr_all.svg")

    official_rows = read_csv(TABLES / "geneval_official_results.csv")
    official_values = [clean_float(row["Overall"]) for row in official_rows if row["Method"] != "Gain"]
    official_labels = [row["Method"] for row in official_rows if row["Method"] != "Gain"]
    bar_svg("GenEval Official Overall", official_labels, official_values, OUT / "geneval_official_overall.svg", "Overall")
    copy_to_paper("geneval_official_overall.svg")


if __name__ == "__main__":
    main()

