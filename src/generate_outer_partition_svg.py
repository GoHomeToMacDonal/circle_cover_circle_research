#!/usr/bin/env python3
"""Generate the 40-point / 25-circle SVG for outer_partition_certificate.md."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from config25 import LABELS, R25, centers25
from outer_partition import LAMBDA, explicit_partition, outer_points

WIDTH, HEIGHT = 1400, 980
CX, CY, SCALE = 485.0, 490.0, 410.0
SVG_NS = "http://www.w3.org/2000/svg"


def xy(point: tuple[float, float] | list[float]) -> tuple[float, float]:
    """Map normalized mathematical coordinates to SVG coordinates."""
    return CX + SCALE * float(point[0]), CY - SCALE * float(point[1])


def line(x1: float, y1: float, x2: float, y2: float, cls: str) -> str:
    return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" class="{cls}"/>'


def covering_circles() -> list[tuple[str, float, float]]:
    centers = centers25() / R25
    return [(label, float(center[0]), float(center[1])) for label, center in zip(LABELS, centers)]


def key_points() -> list[tuple[int, str, float, float, str]]:
    points, labels = outer_points()
    # Section 7: the first eight B_k groups are disjoint and contain 40 points.
    indices = [index for group in explicit_partition()[:8] for index in group]
    if len(indices) != 40 or len(set(indices)) != 40:
        raise AssertionError("the eight B_k groups must contain 40 distinct points")

    result = []
    for index in indices:
        label = labels[index]
        family = (
            "xminus" if label.startswith("X-") else
            "xplus" if label.startswith("X+") else
            "q" if label.startswith("Q") else
            "yb"
        )
        result.append((index, label, float(points[index, 0]), float(points[index, 1]), family))
    return sorted(result)


def build_svg() -> str:
    circles = covering_circles()
    points = key_points()
    all_points, labels = outer_points()
    label_to_point = {
        label: (float(point[0]), float(point[1]))
        for label, point in zip(labels, all_points)
    }

    out: list[str] = []
    add = out.append
    add('<?xml version="1.0" encoding="UTF-8"?>')
    add(f'<svg xmlns="{SVG_NS}" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc" data-covering-circles="25" data-key-points="40">')
    add('<title id="title">外层分拆证书：40 个关键点与 25 个覆盖圆</title>')
    add('<desc id="desc">目标单位圆盘内的 25 个半径 lambda 覆盖圆，以及文档第 7 节八个 B_k 组中的 40 个关键点。鼠标悬停关键点可查看名称。</desc>')
    add('<metadata>normalized-scale; lambda=1/(sqrt(6)+sqrt(3)); covering-circles=25; key-points=40</metadata>')
    add('''<defs>
      <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
        <feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity="0.12"/>
      </filter>
      <style>
        svg { background: #f7f8fa; color: #172033; font-family: Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif; }
        .plot-bg { fill: #fff; stroke: #d8dee9; stroke-width: 1.5; filter: url(#shadow); }
        .target-disk { fill: #f8fafc; stroke: #172033; stroke-width: 3; }
        .axis { stroke: #94a3b8; stroke-width: 1; stroke-dasharray: 5 7; opacity: .45; }
        .covering-circle { fill: none; stroke-width: 1.8; vector-effect: non-scaling-stroke; }
        .circle-O { fill: #64748b; stroke: #475569; stroke-dasharray: 4 4; }
        .circle-A { fill: #14b8a6; stroke: #0f766e; }
        .circle-B { fill: #3b82f6; stroke: #2563eb; stroke-width: 2.3; }
        .circle-C { fill: #f59e0b; stroke: #d97706; stroke-width: 2.3; }
        .center-dot { stroke: #fff; stroke-width: 1.2; }
        .center-O { fill: #475569; } .center-A { fill: #0f766e; }
        .center-B { fill: #2563eb; } .center-C { fill: #d97706; }
        .center-label { font-size: 11px; font-weight: 650; text-anchor: middle; fill: #27364a; stroke: none; }
        .membership { stroke: #2563eb; stroke-width: 1.15; opacity: .20; }
        .key-point { stroke: #fff; stroke-width: 1.8; vector-effect: non-scaling-stroke; cursor: help; }
        .point-xminus { fill: #e11d48; } .point-xplus { fill: #9333ea; }
        .point-q { fill: #16a34a; } .point-yb { fill: #0369a1; }
        .point-node:hover .key-point { r: 7.5px; stroke: #111827; stroke-width: 2.2; }
        .point-label { display: none; pointer-events: none; font-size: 13px; font-weight: 750; fill: #111827;
                       paint-order: stroke; stroke: #fff; stroke-width: 4px; }
        .point-node:hover .point-label { display: block; }
        .panel { fill: #fff; stroke: #d8dee9; stroke-width: 1.5; filter: url(#shadow); }
        .title { font-size: 25px; font-weight: 760; fill: #101827; }
        .subtitle { font-size: 14px; fill: #526176; }
        .section { font-size: 16px; font-weight: 730; fill: #172033; }
        .body { font-size: 14px; fill: #42526a; }
        .small { font-size: 12px; fill: #64748b; }
        .count { font-size: 23px; font-weight: 780; }
        .legend-line { stroke-width: 3; }
        .formula { font-family: "STIX Two Math", "Times New Roman", serif; font-size: 15px; fill: #26364a; }
      </style>
    </defs>''')

    # Main drawing area.
    add('<rect class="plot-bg" x="24" y="24" width="922" height="932" rx="18"/>')
    add(f'<circle class="target-disk" cx="{CX}" cy="{CY}" r="{SCALE}"/>')
    add(line(CX - SCALE, CY, CX + SCALE, CY, "axis"))
    add(line(CX, CY - SCALE, CX, CY + SCALE, "axis"))
    add('<text x="58" y="58" class="small">归一化目标圆盘  |p| ≤ 1</text>')

    # Membership lines: each of the eight B_k disks to its five certificate points.
    add('<g id="B-memberships" aria-label="八个 B_k 组的点归属连线">')
    circle_map = {label: (x, y) for label, x, y in circles}
    point_labels = outer_points()[1]
    for k, group in enumerate(explicit_partition()[:8]):
        bx, by = xy(circle_map[f"B{k}"])
        for point_index in group:
            px, py = xy(label_to_point[point_labels[point_index]])
            add(line(bx, by, px, py, "membership"))
    add('</g>')

    # Explicit presentation attributes complement CSS for older SVG renderers.
    circle_color = {"O": "#475569", "A": "#0f766e", "B": "#2563eb", "C": "#d97706"}
    point_color = {"xminus": "#e11d48", "xplus": "#9333ea", "q": "#16a34a", "yb": "#0369a1"}

    # Exactly 25 semantic covering-circle elements.
    add('<g id="covering-circles" aria-label="25 个覆盖圆">')
    radius = SCALE * LAMBDA
    for label, x, y in circles:
        sx, sy = xy((x, y))
        family = label[0] if label != "O" else "O"
        dash = ' stroke-dasharray="4 4"' if family == "O" else ""
        width = "2.3" if family in {"B", "C"} else "1.8"
        color = circle_color[family]
        add(f'<g><title>{escape(label)}：圆心 ({x:.6f}, {y:.6f})，半径 λ={LAMBDA:.9f}</title><circle class="covering-circle circle-{family}" data-circle="{escape(label)}" cx="{sx:.3f}" cy="{sy:.3f}" r="{radius:.3f}" fill="none" stroke="{color}" stroke-width="{width}"{dash}/></g>')
    add('</g>')

    # Circle centers and compact labels.
    add('<g id="circle-centers" aria-label="覆盖圆圆心">')
    for label, x, y in circles:
        sx, sy = xy((x, y))
        family = label[0] if label != "O" else "O"
        add(f'<circle class="center-dot center-{family}" cx="{sx:.3f}" cy="{sy:.3f}" r="3.5" fill="{circle_color[family]}" stroke="#fff" stroke-width="1.2"/>')
        dy = -7 if y >= 0 else 14
        add(f'<text class="center-label" x="{sx:.3f}" y="{sy + dy:.3f}">{escape(label)}</text>')
    add('</g>')

    # Exactly 40 semantic point markers. Labels appear on hover in a browser.
    add('<g id="key-points" aria-label="40 个关键点">')
    for _index, label, x, y, family in points:
        sx, sy = xy((x, y))
        anchor = "start" if x >= 0 else "end"
        dx = 10 if x >= 0 else -10
        dy = -9 if y >= 0 else 16
        add(f'<g class="point-node" data-point="{escape(label)}"><title>{escape(label)} = ({x:.6f}, {y:.6f})</title>')
        add(f'<circle class="key-point point-{family}" cx="{sx:.3f}" cy="{sy:.3f}" r="5.2" fill="{point_color[family]}" stroke="#fff" stroke-width="1.8" data-x="{x:.9f}" data-y="{y:.9f}"/>')
        add(f'<text class="point-label" x="{sx + dx:.3f}" y="{sy + dy:.3f}" text-anchor="{anchor}">{escape(label)}</text>')
        add('</g>')
    add('</g>')

    # Right-side explanation panel.
    add('<rect class="panel" x="970" y="24" width="406" height="932" rx="18"/>')
    add('<text class="title" x="1002" y="70">40 个关键点 / 25 个圆</text>')
    add('<text class="subtitle" x="1002" y="98">outer_partition_certificate · 第 7 节</text>')
    add('<text class="body" x="1002" y="130">八个 Bₖ 显式组的并集；蓝色细线表示归属。</text>')

    add('<text class="section" x="1002" y="178">关键点（共 40）</text>')
    point_legend = [
        ("#e11d48", "X⁻", "8", "单位圆周交点"),
        ("#9333ea", "X⁺", "8", "单位圆周交点"),
        ("#16a34a", "Q", "16", "j mod 3 ∈ {0, 2}"),
        ("#0369a1", "Yᴮ", "8", "Bₖ 的径向最内点"),
    ]
    y0 = 211
    for i, (color, name, count, note) in enumerate(point_legend):
        y = y0 + i * 42
        add(f'<path d="M1012 {y}a6 6 0 1 0 12 0a6 6 0 1 0-12 0" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        add(f'<text class="body" x="1038" y="{y + 5}"><tspan font-weight="730">{name}</tspan><tspan x="1092">× {count}</tspan><tspan x="1144" class="small">{note}</tspan></text>')

    add('<text class="section" x="1002" y="402">覆盖圆（共 25）</text>')
    circle_legend = [
        ("#475569", "O", "1", "中心层"),
        ("#0f766e", "Aₖ", "8", "内层"),
        ("#2563eb", "Bₖ", "8", "外层 / 覆盖这 40 点"),
        ("#d97706", "Cₖ", "8", "外层"),
    ]
    for i, (color, name, count, note) in enumerate(circle_legend):
        y = 436 + i * 42
        add(f'<path class="legend-line" d="M1007 {y}h24" stroke="{color}"/>')
        add(f'<text class="body" x="1044" y="{y + 5}"><tspan font-weight="730">{name}</tspan><tspan x="1092">× {count}</tspan><tspan x="1144" class="small">{note}</tspan></text>')

    add('<text class="section" x="1002" y="624">尺度与定义</text>')
    add('<text class="formula" x="1002" y="658">λ = 1 / (√6 + √3) ≈ 0.239146312</text>')
    add('<text class="body" x="1002" y="690">目标圆半径：1</text>')
    add('<text class="body" x="1002" y="716">25 个覆盖圆半径：λ</text>')
    add('<text class="body" x="1002" y="742">Bₖ 组：{Xₖ⁻, Xₖ₋₁⁺, Q₃ₖ₋₁, Q₃ₖ, Yₖᴮ}</text>')

    add('<text class="section" x="1002" y="798">观察提示</text>')
    add('<text class="body" x="1002" y="830">• 在浏览器中悬停彩色点可显示名称和坐标。</text>')
    add('<text class="body" x="1002" y="856">• X⁻/X⁺ 位于黑色单位圆周上，彼此很接近。</text>')
    add('<text class="body" x="1002" y="882">• Q 点按 15° 半步相位均匀分布。</text>')
    add('<text class="small" x="1002" y="925">未绘制：其余 8 个 Q₃ₖ₊₁ 与 8 个 Yᶜ（属于 Cₖ 组）。</text>')

    add('</svg>')
    return "\n".join(out) + "\n"


def validate(svg_text: str) -> tuple[int, int]:
    root = ET.fromstring(svg_text)
    circles = root.findall(f".//{{{SVG_NS}}}circle")
    covering = [e for e in circles if "covering-circle" in e.get("class", "").split()]
    points = [e for e in circles if "key-point" in e.get("class", "").split()]
    if len(covering) != 25:
        raise AssertionError(f"expected 25 covering circles, got {len(covering)}")
    if len(points) != 40:
        raise AssertionError(f"expected 40 key points, got {len(points)}")
    if len({e.get("data-circle") for e in covering}) != 25:
        raise AssertionError("covering-circle labels are not unique")
    parent_points = root.findall(f".//{{{SVG_NS}}}g[@class='point-node']")
    if len({e.get("data-point") for e in parent_points}) != 40:
        raise AssertionError("point labels are not unique")
    return len(covering), len(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/outer_partition_40_points_25_circles.svg"),
    )
    args = parser.parse_args()
    svg = build_svg()
    circle_count, point_count = validate(svg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output} ({circle_count} covering circles, {point_count} key points)")


if __name__ == "__main__":
    main()
