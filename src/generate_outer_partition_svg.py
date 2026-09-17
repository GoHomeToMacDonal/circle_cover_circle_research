#!/usr/bin/env python3
"""Generate the outer-partition SVG with the 13-point inner witness overlay."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from config25 import LABELS, R25, centers25
from outer_partition import LAMBDA, explicit_partition, outer_points

WIDTH, HEIGHT = 1400, 1140
CX, CY, SCALE = 485.0, 490.0, 410.0
SVG_NS = "http://www.w3.org/2000/svg"
CRITICAL_OCTAGON_RADIUS = math.sqrt((4.0 - 2.0 * math.sqrt(2.0)) / 3.0)


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


def inner_witness_points() -> list[tuple[str, float, float, str]]:
    """Return the critical octagon, its centre, and four isolating unit-circle points."""
    points: list[tuple[str, float, float, str]] = [("W-O", 0.0, 0.0, "center")]
    for k in range(8):
        angle = math.pi / 8.0 + k * math.pi / 4.0
        points.append((
            f"W-V{k}",
            CRITICAL_OCTAGON_RADIUS * math.cos(angle),
            CRITICAL_OCTAGON_RADIUS * math.sin(angle),
            "octagon",
        ))
    for k in range(4):
        angle = k * math.pi / 2.0
        points.append((f"W-Q{k}", math.cos(angle), math.sin(angle), "added"))
    return points


def build_svg() -> str:
    circles = covering_circles()
    points = key_points()
    witness_points = inner_witness_points()
    all_points, labels = outer_points()
    label_to_point = {
        label: (float(point[0]), float(point[1]))
        for label, point in zip(labels, all_points)
    }

    out: list[str] = []
    add = out.append
    add('<?xml version="1.0" encoding="UTF-8"?>')
    add(f'<svg xmlns="{SVG_NS}" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc" data-covering-circles="25" data-key-points="40" data-inner-witness-points="13">')
    add('<title id="title">外层分拆证书与内层 13 点见证</title>')
    add('<desc id="desc">保留 25 个半径 lambda 覆盖圆和原有 40 个关键点，并叠加临界正八边形、中心点及四个单位圆周新增点。鼠标悬停点可查看名称。</desc>')
    add('<metadata>normalized-scale; lambda=1/(sqrt(6)+sqrt(3)); covering-circles=25; key-points=40; inner-witness-points=13</metadata>')
    add('''<defs>
      <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
        <feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity="0.12"/>
      </filter>
      <style>
        svg { background: #f7f8fa; color: #172033; font-family: Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif; }
        .plot-bg { fill: #fff; stroke: #d8dee9; stroke-width: 1.5; filter: url(#shadow); }
        .target-disk { fill: none; stroke: #172033; stroke-width: 3; }
        .axis { stroke: #94a3b8; stroke-width: 1; stroke-dasharray: 5 7; opacity: .45; }
        .covering-circle { fill: none; stroke-width: 1.8; vector-effect: non-scaling-stroke; }
        .circle-O { fill: none; stroke: #475569; stroke-dasharray: 4 4; }
        .circle-A { fill: none; stroke: #0f766e; }
        .circle-B { fill: none; stroke: #2563eb; stroke-width: 2.3; }
        .circle-C { fill: none; stroke: #d97706; stroke-width: 2.3; }
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
        .point-node:hover .point-label, .witness-node:hover .point-label { display: block; }
        .witness-octagon { fill: none; stroke: #dc2626; stroke-width: 2.4; stroke-dasharray: 7 5; }
        .witness-point { stroke: #fff; stroke-width: 2; cursor: help; vector-effect: non-scaling-stroke; }
        .witness-center { fill: #111827; } .witness-octagon-point { fill: #dc2626; }
        .witness-added { fill: #0891b2; }
        .witness-node:hover .witness-point { stroke: #111827; stroke-width: 2.6; }
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
    add('<rect class="plot-bg" x="24" y="24" width="922" height="1092" rx="18"/>')
    add(f'<circle class="target-disk" cx="{CX}" cy="{CY}" r="{SCALE}" fill="none"/>')
    add(line(CX - SCALE, CY, CX + SCALE, CY, "axis"))
    add(line(CX, CY - SCALE, CX, CY + SCALE, "axis"))
    add('<text x="58" y="58" class="small">归一化目标圆盘  |p| ≤ 1</text>')

    # Critical regular octagon outline; point markers are drawn above the original layers.
    octagon_svg_points = [xy((x, y)) for _label, x, y, family in witness_points if family == "octagon"]
    octagon_coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in octagon_svg_points)
    add('<g id="inner-witness-geometry" aria-label="临界正八边形轮廓">')
    add(f'<polygon class="witness-octagon" points="{octagon_coords}" fill="none"/>')
    add('</g>')

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

    # Thirteen-point inner witness: centre, eight critical octagon vertices, four added points.
    witness_color = {"center": "#111827", "octagon": "#dc2626", "added": "#0891b2"}
    witness_radius = {"center": 6.0, "octagon": 5.8, "added": 6.6}
    add('<g id="inner-witness-points" aria-label="13 个内层下界见证点">')
    for label, x, y, family in witness_points:
        sx, sy = xy((x, y))
        anchor = "start" if x >= 0 else "end"
        dx = 11 if x >= 0 else -11
        dy = -10 if y >= 0 else 17
        css_family = "octagon-point" if family == "octagon" else family
        add(f'<g class="witness-node" data-witness-point="{escape(label)}"><title>{escape(label)} = ({x:.6f}, {y:.6f})</title>')
        add(f'<circle class="witness-point witness-{css_family}" cx="{sx:.3f}" cy="{sy:.3f}" r="{witness_radius[family]:.1f}" fill="{witness_color[family]}" stroke="#fff" stroke-width="2" data-x="{x:.9f}" data-y="{y:.9f}"/>')
        add(f'<text class="point-label" x="{sx + dx:.3f}" y="{sy + dy:.3f}" text-anchor="{anchor}">{escape(label)}</text>')
        add('</g>')
    add('</g>')

    # Right-side explanation panel.
    add('<rect class="panel" x="970" y="24" width="406" height="1092" rx="18"/>')
    add('<text class="title" x="1002" y="70">40 关键点 + 13 见证点</text>')
    add('<text class="subtitle" x="1002" y="98">outer_partition_certificate · 保留原图并叠加内层见证</text>')
    add('<text class="body" x="1002" y="130">原有 25 圆、40 点和 Bₖ 归属线均保留。</text>')

    add('<text class="section" x="1002" y="178">原有关键点（共 40）</text>')
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

    add('<text class="section" x="1002" y="402">覆盖圆（共 25，均无填充）</text>')
    circle_legend = [
        ("#475569", "O", "1", "中心层"),
        ("#0f766e", "Aₖ", "8", "内层"),
        ("#2563eb", "Bₖ", "8", "外层 / 覆盖原 40 点"),
        ("#d97706", "Cₖ", "8", "外层"),
    ]
    for i, (color, name, count, note) in enumerate(circle_legend):
        y = 436 + i * 42
        add(f'<path class="legend-line" d="M1007 {y}h24" stroke="{color}"/>')
        add(f'<text class="body" x="1044" y="{y + 5}"><tspan font-weight="730">{name}</tspan><tspan x="1092">× {count}</tspan><tspan x="1144" class="small">{note}</tspan></text>')

    add('<text class="section" x="1002" y="624">新增内层见证点（共 13）</text>')
    witness_legend = [
        ("#dc2626", "Vₖ", "8", "临界正八边形顶点"),
        ("#111827", "O*", "1", "正八边形中心"),
        ("#0891b2", "Q*", "4", "单位圆坐标轴点"),
    ]
    for i, (color, name, count, note) in enumerate(witness_legend):
        y = 658 + i * 38
        add(f'<path d="M1012 {y}a6 6 0 1 0 12 0a6 6 0 1 0-12 0" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        add(f'<text class="body" x="1038" y="{y + 5}"><tspan font-weight="730">{name}</tspan><tspan x="1092">× {count}</tspan><tspan x="1144" class="small">{note}</tspan></text>')
    add('<text class="formula" x="1002" y="776">兼容图：C₈ ⊔ 5K₁　⇒　团覆盖数 = 9</text>')

    add('<text class="section" x="1002" y="824">尺度与定义</text>')
    add('<text class="formula" x="1002" y="858">λ = 1 / (√6 + √3) ≈ 0.239146312</text>')
    add('<text class="formula" x="1002" y="886">ρ* = √((4 − 2√2)/3) ≈ 0.624919428</text>')
    add('<text class="body" x="1002" y="914">临界八边形边长：2λ（红色虚线）</text>')
    add('<text class="body" x="1002" y="940">目标圆及 25 个覆盖圆：只有描边，无填充</text>')

    add('<text class="section" x="1002" y="988">观察提示</text>')
    add('<text class="body" x="1002" y="1020">• 悬停彩色点可显示名称和坐标。</text>')
    add('<text class="body" x="1002" y="1046">• 红色相邻顶点距离恰为 2λ。</text>')
    add('<text class="body" x="1002" y="1072">• 青色四点与其余点距离严格大于 2λ。</text>')
    add('<text class="small" x="1002" y="1102">原证书未绘制的 8 个 Q₃ₖ₊₁ 与 8 个 Yᶜ 仍保持未绘制。</text>')

    add('<text class="section" x="58" y="985">内层 13 点见证</text>')
    add('<text class="formula" x="58" y="1018">边长 s* = 2λ；外接半径 ρ* = √((4 − 2√2)/3)</text>')
    add('<text class="body" x="58" y="1048">红色八边形顶点形成 C₈；中心与四个青色点均为兼容图孤立点。</text>')
    add('<text class="body" x="58" y="1076">所以任何半径 λ 的圆覆盖需要至少 4 + 5 = 9 个圆。</text>')

    add('</svg>')
    return "\n".join(out) + "\n"


def validate(svg_text: str) -> tuple[int, int, int]:
    root = ET.fromstring(svg_text)
    circles = root.findall(f".//{{{SVG_NS}}}circle")
    covering = [e for e in circles if "covering-circle" in e.get("class", "").split()]
    points = [e for e in circles if "key-point" in e.get("class", "").split()]
    witnesses = [e for e in circles if "witness-point" in e.get("class", "").split()]
    if len(covering) != 25:
        raise AssertionError(f"expected 25 covering circles, got {len(covering)}")
    if len(points) != 40:
        raise AssertionError(f"expected 40 key points, got {len(points)}")
    if len(witnesses) != 13:
        raise AssertionError(f"expected 13 inner witness points, got {len(witnesses)}")
    if any(e.get("fill") != "none" for e in covering):
        raise AssertionError("every covering circle must have fill=none")
    target = root.find(f".//{{{SVG_NS}}}circle[@class='target-disk']")
    if target is None or target.get("fill") != "none":
        raise AssertionError("target circle must have fill=none")
    if len({e.get("data-circle") for e in covering}) != 25:
        raise AssertionError("covering-circle labels are not unique")
    parent_points = root.findall(f".//{{{SVG_NS}}}g[@class='point-node']")
    if len({e.get("data-point") for e in parent_points}) != 40:
        raise AssertionError("point labels are not unique")
    witness_nodes = root.findall(f".//{{{SVG_NS}}}g[@class='witness-node']")
    if len({e.get("data-witness-point") for e in witness_nodes}) != 13:
        raise AssertionError("witness-point labels are not unique")
    return len(covering), len(points), len(witnesses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/outer_partition_40_points_25_circles.svg"),
    )
    args = parser.parse_args()
    svg = build_svg()
    circle_count, point_count, witness_count = validate(svg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(
        f"wrote {args.output} ({circle_count} covering circles, "
        f"{point_count} key points, {witness_count} inner witness points)"
    )


if __name__ == "__main__":
    main()
