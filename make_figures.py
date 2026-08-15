from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="3"/>'


def main() -> None:
    results = pd.read_csv(ROOT / "results.csv")
    summary = pd.read_csv(ROOT / "summary.csv")
    FIGURES.mkdir(exist_ok=True)

    width, height = 980, 560
    left, top, plot_w, plot_h = 80, 70, 560, 380
    rounds = sorted(results["round"].unique())
    max_round = max(rounds)
    min_auc, max_auc = 0.0, 1.0
    colors = {
        "fedavg": "#2563eb",
        "topk": "#16a34a",
        "topk_ef": "#65a30d",
        "topk_sign": "#dc2626",
        "topk_sign_ef": "#9333ea",
    }

    lines = []
    for mode, group in results.groupby("mode"):
        points = []
        for _, row in group.sort_values("round").iterrows():
            x = left + (row["round"] - 1) / max(1, max_round - 1) * plot_w
            y = top + (max_auc - row["auc"]) / (max_auc - min_auc) * plot_h
            points.append((x, y))
        lines.append(_polyline(points, colors.get(mode, "#111827")))
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors.get(mode, "#111827")}"/>')

    legend = []
    for idx, mode in enumerate(colors):
        y = top + idx * 28
        legend.append(f'<rect x="690" y="{y}" width="14" height="14" fill="{colors[mode]}"/>')
        legend.append(f'<text x="712" y="{y + 12}" class="small">{mode}</text>')

    bars = []
    for idx, row in summary.sort_values("communication_ratio", ascending=False).iterrows():
        y = 285 + idx * 38
        bar_w = max(2, row["communication_ratio"] * 220)
        color = colors.get(row["mode"], "#111827")
        bars.append(f'<text x="690" y="{y - 6}" class="small">{row["mode"]}</text>')
        bars.append(f'<rect x="690" y="{y}" width="{bar_w:.1f}" height="16" fill="{color}"/>')
        bars.append(f'<text x="{700 + bar_w:.1f}" y="{y + 13}" class="tiny">{row["communication_ratio"] * 100:.3f}%</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 24px Arial, sans-serif; fill: #111827; }}
  .label {{ font: 600 14px Arial, sans-serif; fill: #374151; }}
  .small {{ font: 13px Arial, sans-serif; fill: #374151; }}
  .tiny {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .axis {{ stroke: #6b7280; stroke-width: 1.5; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="40" y="38" class="title">Pathology Federated Learning: AUC vs Communication</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
<text x="76" y="54" text-anchor="middle" class="label">AUC</text>
<text x="{left + plot_w / 2}" y="{top + plot_h + 45}" text-anchor="middle" class="label">Federated Round</text>
"""
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1 - tick) * plot_h
        svg += f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>\n'
        svg += f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tiny">{tick:.2f}</text>\n'
    for round_id in rounds:
        x = left + (round_id - 1) / max(1, max_round - 1) * plot_w
        svg += f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tiny">{round_id}</text>\n'
    svg += "\n".join(lines)
    svg += '\n<text x="690" y="54" class="label">Methods</text>\n'
    svg += "\n".join(legend)
    svg += '\n<text x="690" y="250" class="label">Final Communication Load</text>\n'
    svg += "\n".join(bars)
    svg += "\n</svg>\n"

    out = FIGURES / "auc_communication.svg"
    out.write_text(svg, encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
