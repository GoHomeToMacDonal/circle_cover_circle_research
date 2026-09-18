"""Generate exploratory shell areas and boundary supercells for the n=25 atlas."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import scipy.integrate as integrate
import sympy as sp

from n25_voronoi_atlas import exact_sign

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATLAS = ROOT / "data" / "n25_voronoi_atlas.json"
DEFAULT_JSON = ROOT / "data" / "n25_shell_supercells.json"
DEFAULT_DOC = ROOT / "docs" / "n25_shell_supercells.md"
EXPECTED_ATLAS_HASH = "cda22f3719de1e20eb100ae011667fbe25952137b82be77c74e8b5726fc92257"
EXPECTED_SCHEMA = "n25-clipped-voronoi-delaunay-atlas-v3"
MP_DIGITS = 90
AREA_TOLERANCE = mp.mpf("1e-60")
ANGLE_TOLERANCE = mp.mpf("1e-70")
POLAR_TOLERANCE = 1e-9


def expr(text):
    return sp.sympify(text)


def exact_point(record):
    return expr(record["coordinate"][0]), expr(record["coordinate"][1])


def mp_value(value):
    return mp.mpf(str(sp.N(value, MP_DIGITS)))


def mp_point(point):
    return mp_value(point[0]), mp_value(point[1])


def nstr(value):
    value = mp.mpf(value)
    if abs(value) < mp.mpf("1e-85"):
        value = mp.mpf(0)
    return mp.nstr(value, 80)


def tolstr(value):
    value = mp.mpf(value)
    if value == AREA_TOLERANCE:
        return "1e-60"
    if value == ANGLE_TOLERANCE:
        return "1e-70"
    return nstr(value)


def qstr(value):
    return str(sp.factor(sp.cancel(sp.expand(value))))


def canonical_sha256(material):
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_atlas_hash(atlas):
    material = copy.deepcopy(atlas)
    material.pop("telemetry", None)
    material.pop("deterministic_sha256", None)
    if isinstance(material.get("candidate_enumeration"), dict):
        material["candidate_enumeration"].pop("telemetry", None)
    return canonical_sha256(material)


def canonical_manifest_hash(payload):
    material = copy.deepcopy(payload)
    material.pop("telemetry", None)
    material.pop("deterministic_manifest_sha256", None)
    return canonical_sha256(material)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def directed_angle(a, b, direction):
    value = mp.atan2(cross(a, b), dot(a, b))
    if direction == "CCW":
        while value <= 0:
            value += 2 * mp.pi
        return value
    if direction == "CW":
        while value >= 0:
            value -= 2 * mp.pi
        return value
    raise ValueError(f"unknown arc direction: {direction}")


def assert_short_arc(angle):
    assert abs(angle) < mp.pi, f"atlas boundary arc is not a short arc: {nstr(angle)}"


def line_intersection_details(a, b, radius):
    d = (b[0] - a[0], b[1] - a[1])
    aa = dot(d, d)
    if aa == 0:
        return mp.mpf(0), {"quadratic_cuts": ["0.0", "1.0"], "pieces": []}
    bb = 2 * dot(a, d)
    cc = dot(a, a) - radius * radius
    discriminant = bb * bb - 4 * aa * cc
    scale = max(mp.mpf(1), abs(bb * bb), abs(4 * aa * cc))
    if discriminant < 0 and abs(discriminant) <= mp.mpf("1e-75") * scale:
        discriminant = mp.mpf(0)
    cuts = [mp.mpf(0), mp.mpf(1)]
    if discriminant >= 0:
        root = mp.sqrt(discriminant)
        for value in ((-bb - root) / (2 * aa), (-bb + root) / (2 * aa)):
            if value > 0 and value < 1:
                cuts.append(value)
    cuts = sorted(set(cuts))
    pieces = []
    area = mp.mpf(0)
    for left, right in zip(cuts, cuts[1:]):
        p = (a[0] + left * d[0], a[1] + left * d[1])
        q = (a[0] + right * d[0], a[1] + right * d[1])
        mid_lambda = (left + right) / 2
        mid = (a[0] + mid_lambda * d[0], a[1] + mid_lambda * d[1])
        inside = dot(mid, mid) <= radius * radius
        if inside:
            primitive = "cross(p,q)/2"
            contribution = cross(p, q) / 2
        else:
            primitive = "t^2*directed_angle(p,q)/2"
            contribution = radius * radius * mp.atan2(cross(p, q), dot(p, q)) / 2
        area += contribution
        pieces.append({
            "lambda_endpoints": [nstr(left), nstr(right)],
            "inside_outside": "inside" if inside else "outside",
            "primitive": primitive,
            "contribution": nstr(contribution),
        })
    return area, {"quadratic_cuts": [nstr(x) for x in cuts], "pieces": pieces}


def segment_radius_exact(a, b):
    d = (sp.expand(b[0] - a[0]), sp.expand(b[1] - a[1]))
    dd = sp.expand(d[0] * d[0] + d[1] * d[1])
    if exact_sign(dd, "segment squared length") == 0:
        value = sp.expand(a[0] * a[0] + a[1] * a[1])
        return value, value, {"lambda_status": "degenerate segment", "high_endpoint_comparison": "equal"}
    numerator = sp.expand(a[0] * d[0] + a[1] * d[1])
    lam = sp.cancel(-numerator / dd)
    lam_sign = exact_sign(lam, "segment projection lambda")
    one_minus_lam_sign = exact_sign(lam - 1, "segment projection lambda minus one")
    if lam_sign <= 0:
        low = sp.expand(a[0] * a[0] + a[1] * a[1])
        lambda_status = "projection at or before first endpoint"
    elif one_minus_lam_sign >= 0:
        low = sp.expand(b[0] * b[0] + b[1] * b[1])
        lambda_status = "projection at or after second endpoint"
    else:
        low = sp.cancel(sp.expand(a[0] * a[0] + a[1] * a[1] - numerator * numerator / dd))
        lambda_status = "projection strictly inside segment"
    a_sq = sp.expand(a[0] * a[0] + a[1] * a[1])
    b_sq = sp.expand(b[0] * b[0] + b[1] * b[1])
    high_sign = exact_sign(a_sq - b_sq, "segment endpoint radial comparison")
    high = a_sq if high_sign >= 0 else b_sq
    return low, high, {
        "lambda_exact": qstr(lam),
        "lambda_sign": lam_sign,
        "lambda_minus_one_sign": one_minus_lam_sign,
        "lambda_status": lambda_status,
        "high_endpoint_comparison": "first endpoint >= second" if high_sign >= 0 else "second endpoint > first",
        "high_endpoint_sign": high_sign,
    }


def role(label):
    return "O" if label == "O" else label[0]


def orbit_record(label):
    return {"group": "D8", "index": 0 if label == "O" else int(label[1:])}


def edge_path(cell, edges, vertices):
    cycle = cell["vertex_cycle"]
    result = []
    for pos, edge_id in enumerate(cell["edge_cycle"]):
        edge = edges[edge_id]
        start_id = cycle[pos]
        end_id = cycle[(pos + 1) % len(cycle)]
        if set(edge["vertices"]) != {start_id, end_id}:
            raise AssertionError(f"edge-cycle incidence mismatch at edge {edge_id}")
        start = exact_point(vertices[start_id])
        end = exact_point(vertices[end_id])
        result.append((edge, start, end, start_id, end_id))
    return result


def edge_audit(edge, start, end, start_id, end_id, radii):
    base = {
        "edge_id": edge["id"],
        "kind": edge["kind"],
        "directed_vertex_ids": [start_id, end_id],
        "exact_endpoint_coordinates": [[qstr(start[0]), qstr(start[1])], [qstr(end[0]), qstr(end[1])]],
        "contributions": [],
    }
    if edge["kind"] == "target-boundary-arc":
        direction = edge["arc_status"]["directed_endpoints"]["direction"]
        angle = directed_angle(mp_point(start), mp_point(end), direction)
        assert_short_arc(angle)
        base["directed_angle"] = nstr(angle)
        base["short_arc_assertion"] = {"abs_angle_lt_pi": True, "passed": abs(angle) < mp.pi}
        base["primitive_type"] = "t^2*DeltaTheta/2"
        for radius in radii:
            contribution = radius * radius * angle / 2
            base["contributions"].append({"radius": nstr(radius), "primitive": "t^2*DeltaTheta/2", "contribution": nstr(contribution)})
    else:
        base["primitive_type"] = "piecewise Green line primitive"
        for radius in radii:
            contribution, details = line_intersection_details(mp_point(start), mp_point(end), radius)
            base["contributions"].append({"radius": nstr(radius), **details, "contribution": nstr(contribution)})
    return base


def recompute_atlas_hash(atlas):
    return canonical_atlas_hash(atlas)


def polar_cross_check(atlas, cells, centers, radii, vertices):
    vertex_angles = []
    for vertex in vertices:
        x, y = mp_point(exact_point(vertex))
        angle = float(mp.atan2(y, x))
        if angle < 0:
            angle += 2 * math.pi
        if 1e-13 < angle < 2 * math.pi - 1e-13:
            vertex_angles.append(angle)
    points = sorted(set(vertex_angles))
    target_radius = float(mp_value(radii[-1]))
    center_float = [(float(mp_value(x)), float(mp_value(y))) for x, y in centers]
    constraints = []
    for site, center in enumerate(centers):
        site_constraints = []
        for other, other_center in enumerate(centers):
            if site == other:
                continue
            ax = 2 * (other_center[0] - center[0])
            ay = 2 * (other_center[1] - center[1])
            bb = other_center[0] ** 2 + other_center[1] ** 2 - center[0] ** 2 - center[1] ** 2
            ax_sign = exact_sign(ax, f"polar constraint ax {site},{other}")
            ay_sign = exact_sign(ay, f"polar constraint ay {site},{other}")
            site_constraints.append((float(mp_value(ax)), float(mp_value(ay)), float(mp_value(bb)), ax_sign, ay_sign))
        constraints.append(site_constraints)

    def radial_interval(theta, site, upper):
        ux, uy = math.cos(theta), math.sin(theta)
        lower, upper_bound = 0.0, min(float(upper), target_radius)
        for ax, ay, bb, _, _ in constraints[site]:
            coefficient = ax * ux + ay * uy
            if coefficient > 1e-14:
                upper_bound = min(upper_bound, bb / coefficient)
            elif coefficient < -1e-14:
                lower = max(lower, bb / coefficient)
            elif bb < -1e-12:
                return 0.0, 0.0
        if upper_bound <= lower or upper_bound <= 0:
            return 0.0, 0.0
        return max(0.0, lower), max(0.0, upper_bound)

    selected = {0, 1, 9, 17}
    result = {}
    max_error = 0.0
    for site in selected:
        cell_result = {}
        for radius in radii:
            radius_float = float(mp_value(radius))

            def integrand(theta):
                lower, upper = radial_interval(theta, site, radius_float)
                return 0.5 * max(0.0, upper * upper - lower * lower)

            value, error = integrate.quad(integrand, 0.0, 2 * math.pi, points=points, epsabs=2e-11, epsrel=2e-11, limit=2000)
            expected = float(mp.mpf(cells[site]["cumulative_areas"][radii.index(radius)]))
            abs_error = abs(value - expected)
            max_error = max(max_error, abs_error)
            cell_result[nstr(radius)] = {
                "polar_value": nstr(value),
                "green_value": nstr(expected),
                "absolute_error": nstr(abs_error),
                "quadrature_error_estimate": nstr(error),
                "tolerance": "1e-9",
                "passed": abs_error <= POLAR_TOLERANCE,
            }
        result[cells[site]["label"]] = cell_result
    return {"method": "independent polar-ray radial intervals with scipy.integrate.quad", "vertex_polar_breakpoints": len(points), "cells": result, "max_absolute_error": nstr(max_error), "tolerance": "1e-9", "passed": max_error <= POLAR_TOLERANCE}


def build(atlas):
    assert atlas["schema"] == EXPECTED_SCHEMA
    assert atlas["status"] == "exploration"
    recomputed_hash = recompute_atlas_hash(atlas)
    assert recomputed_hash == EXPECTED_ATLAS_HASH
    assert len(atlas["cells"]) == 25
    assert len(atlas["vertices"]) == 48
    assert len(atlas["internal_voronoi_edges"]) == 56
    assert len(atlas["target_boundary_edges"]) == 16
    vertices = atlas["vertices"]
    edges = {edge["id"]: edge for edge in atlas["internal_voronoi_edges"] + atlas["target_boundary_edges"]}
    centers = [tuple(expr(x) for x in pair) for pair in atlas["candidate"]["centers_exact"]]
    radii = [sp.Integer(1), 1 + sp.sqrt(2), sp.sqrt(5 + 2 * sp.sqrt(2)), sp.sqrt(9 + 6 * sp.sqrt(2))]
    radius_sq = [sp.expand(x * x) for x in radii]
    radius_mp = [mp_value(x) for x in radii]
    labels = [cell["label"] for cell in atlas["cells"]]
    cell_records = []
    cell_area_cache = {}
    for cell in atlas["cells"]:
        path = edge_path(cell, edges, vertices)
        cumulative = []
        boundary_angle = mp.mpf(0)
        boundary_arc_ids = []
        neighbors = set()
        green_edges = []
        for edge, start_exact, end_exact, start_id, end_id in path:
            green_edges.append(edge_audit(edge, start_exact, end_exact, start_id, end_id, radius_mp))
            if edge["kind"] == "target-boundary-arc":
                direction = edge["arc_status"]["directed_endpoints"]["direction"]
                angle = directed_angle(mp_point(start_exact), mp_point(end_exact), direction)
                assert_short_arc(angle)
                boundary_angle += angle
                boundary_arc_ids.append(edge["id"])
            else:
                neighbors.update(site for site in edge["sites"] if site != cell["site"])
        for radius in radius_mp:
            total = mp.mpf(0)
            for edge, start_exact, end_exact, _, _ in path:
                if edge["kind"] == "target-boundary-arc":
                    direction = edge["arc_status"]["directed_endpoints"]["direction"]
                    angle = directed_angle(mp_point(start_exact), mp_point(end_exact), direction)
                    assert_short_arc(angle)
                    total += radius * radius * angle / 2
                else:
                    contribution, _ = line_intersection_details(mp_point(start_exact), mp_point(end_exact), radius)
                    total += contribution
            cumulative.append(total)
        shells = [cumulative[0], cumulative[1] - cumulative[0], cumulative[2] - cumulative[1], cumulative[3] - cumulative[2]]
        radial_branches = []
        low_sq = None
        high_sq = None
        for edge, start_exact, end_exact, _, _ in path:
            if edge["kind"] == "target-boundary-arc":
                candidate_low = candidate_high = radius_sq[3]
                branch = {"edge_id": edge["id"], "kind": edge["kind"], "squared_exact": qstr(radius_sq[3]), "selection": "target boundary arc has constant radius R^2"}
            else:
                candidate_low, candidate_high, status = segment_radius_exact(start_exact, end_exact)
                branch = {"edge_id": edge["id"], "kind": edge["kind"], "squared_exact": [qstr(candidate_low), qstr(candidate_high)], "selection": status}
            radial_branches.append(branch)
            if low_sq is None or exact_sign(candidate_low - low_sq, f"cell {cell['site']} radial minimum") < 0:
                low_sq = candidate_low
            if high_sq is None or exact_sign(candidate_high - high_sq, f"cell {cell['site']} radial maximum") > 0:
                high_sq = candidate_high
        if cell["site"] == 0:
            low_sq = sp.Integer(0)
            radial_origin_override = "cell site O contains the origin exactly"
        else:
            radial_origin_override = "not applicable"
        layer_signature = [item["layer"] for item in cell["layer_crossings"]["layers"] if item["crosses"]]
        cell_area_cache[cell["site"]] = {"cumulative": cumulative, "shells": shells}
        cell_records.append({
            "site": cell["site"], "label": cell["label"], "role": role(cell["label"]), "orbit": orbit_record(cell["label"]),
            "total_area": nstr(cumulative[3]), "shell_areas": [nstr(x) for x in shells], "cumulative_areas": [nstr(x) for x in cumulative],
            "boundary_responsibility_angle": {"arc_ids": boundary_arc_ids, "expression": "0" if not boundary_arc_ids else " + ".join(f"DeltaTheta(edge {edge_id})" for edge_id in boundary_arc_ids), "value": nstr(boundary_angle), "status": "high-precision numerical evaluation/display; not an interval"},
            "vertex_cycle": list(cell["vertex_cycle"]), "edge_cycle": list(cell["edge_cycle"]), "vertex_count": cell["vertex_count"],
            "adjacent_sites": sorted(neighbors), "adjacent_roles": sorted({role(labels[x]) for x in neighbors}),
            "layer_signature": layer_signature,
            "radial_range": {"squared_exact": [qstr(low_sq), qstr(high_sq)], "numeric": [nstr(mp.sqrt(mp_value(low_sq))), nstr(mp.sqrt(mp_value(high_sq)))], "selection_verification": {"branch_records": radial_branches, "origin_override": radial_origin_override, "all_min_max_selections_use_exact_sign": True}},
            "green_primitive_audit": {"radii": [nstr(x) for x in radius_mp], "directed_edges": green_edges},
            "formula_status": {"area_formula": "audited directed Green primitives per edge and per radius", "verification": "high-precision numerical evaluation/display; not an interval"},
        })
    shell_totals = [sum((cell_area_cache[i]["shells"][j] for i in range(25)), mp.mpf(0)) for j in range(4)]
    cumulative_totals = [sum((cell_area_cache[i]["cumulative"][j] for i in range(25)), mp.mpf(0)) for j in range(4)]
    theoretical_shells = [mp.pi * (radius_mp[j] ** 2 - (mp.mpf(0) if j == 0 else radius_mp[j - 1] ** 2)) for j in range(4)]
    conservation = {"shells": [], "cumulative": [], "total_area": {}, "responsibility_angle": {}}
    for index in range(4):
        error = abs(shell_totals[index] - theoretical_shells[index])
        conservation["shells"].append({"shell": f"Omega{index}", "computed": nstr(shell_totals[index]), "theoretical": nstr(theoretical_shells[index]), "absolute_error": nstr(error), "tolerance": tolstr(AREA_TOLERANCE), "passed": error <= AREA_TOLERANCE})
        error_c = abs(cumulative_totals[index] - mp.pi * radius_mp[index] ** 2)
        conservation["cumulative"].append({"radius": f"t{index}", "computed": nstr(cumulative_totals[index]), "theoretical": nstr(mp.pi * radius_mp[index] ** 2), "absolute_error": nstr(error_c), "tolerance": tolstr(AREA_TOLERANCE), "passed": error_c <= AREA_TOLERANCE})
    total_error = abs(sum(shell_totals) - mp.pi * radius_mp[3] ** 2)
    conservation["total_area"] = {"computed": nstr(sum(shell_totals)), "theoretical": nstr(mp.pi * radius_mp[3] ** 2), "absolute_error": nstr(total_error), "tolerance": tolstr(AREA_TOLERANCE), "passed": total_error <= AREA_TOLERANCE}
    total_angle = sum((mp.mpf(item["boundary_responsibility_angle"]["value"]) for item in cell_records), mp.mpf(0))
    angle_error = abs(total_angle - 2 * mp.pi)
    conservation["responsibility_angle"] = {"computed": nstr(total_angle), "theoretical": nstr(2 * mp.pi), "absolute_error": nstr(angle_error), "tolerance": tolstr(ANGLE_TOLERANCE), "passed": angle_error <= ANGLE_TOLERANCE}
    boundary_edges = {edge["id"]: edge for edge in atlas["target_boundary_edges"]}
    internal_edges = {edge["id"]: edge for edge in atlas["internal_voronoi_edges"]}
    by_site = {item["site"]: item for item in cell_records}

    def site(label):
        return labels.index(label)

    def arc_for(site_id):
        return next(edge for edge in boundary_edges.values() if edge["sites"] == [site_id])

    def seam_for(left, right):
        return next(edge for edge in internal_edges.values() if set(edge["sites"]) == {left, right})

    def window(name, first, second, pattern):
        left_arc, right_arc = arc_for(first), arc_for(second)
        seam = seam_for(first, second)
        p, q = exact_point(vertices[seam["vertices"][0]]), exact_point(vertices[seam["vertices"][1]])
        length_sq = sp.expand((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)
        low_sq, high_sq, status = segment_radius_exact(p, q)
        angle = mp.mpf(by_site[first]["boundary_responsibility_angle"]["value"]) + mp.mpf(by_site[second]["boundary_responsibility_angle"]["value"])
        shell = [mp.mpf(by_site[first]["shell_areas"][j]) + mp.mpf(by_site[second]["shell_areas"][j]) for j in range(4)]
        return {"id": name, "ordered_sites": [first, second], "ordered_labels": [labels[first], labels[second]], "orientation": "CCW", "common_internal_seam": {"id": seam["id"], "vertices": seam["vertices"], "length": {"squared_exact": qstr(length_sq), "value": nstr(mp.sqrt(mp_value(length_sq)))}, "radial_range": {"squared_exact": [qstr(low_sq), qstr(high_sq)], "value": [nstr(mp.sqrt(mp_value(low_sq))), nstr(mp.sqrt(mp_value(high_sq)))], "selection_verification": status}}, "joint_shell_areas": [nstr(x) for x in shell], "joint_total_area": nstr(sum(shell)), "joint_responsibility_angle": {"value": nstr(angle), "target": nstr(mp.pi / 4), "absolute_error": nstr(abs(angle - mp.pi / 4)), "tolerance": tolstr(ANGLE_TOLERANCE), "passed": abs(angle - mp.pi / 4) <= ANGLE_TOLERANCE}, "boundary": {"start": left_arc["arc_status"]["directed_endpoints"]["from"], "switch": left_arc["arc_status"]["directed_endpoints"]["to"], "end": right_arc["arc_status"]["directed_endpoints"]["to"]}, "external_composition_signature": {"pattern": pattern, "arc_ids": [left_arc["id"], right_arc["id"]], "roles": [role(labels[first]), role(labels[second])], "boundary_arc_directions": [left_arc["arc_status"]["directed_endpoints"]["direction"], right_arc["arc_status"]["directed_endpoints"]["direction"]]}}

    windows = []
    for k in range(8):
        windows.append(window(f"BC_{k}", 9 + k, 17 + k, "B_k union C_k"))
        windows.append(window(f"CB_{k}", 17 + k, 9 + ((k + 1) % 8), "C_k union B_(k+1 mod 8)"))
    matchings = [{"name": "BC", "pairs": [f"BC_{k}" for k in range(8)], "pair_count": 8, "perfect_matching": True}, {"name": "CB", "pairs": [f"CB_{k}" for k in range(8)], "pair_count": 8, "perfect_matching": True}]
    role_summary = {}
    expected = {"O": 1, "A": 8, "B": 8, "C": 8}
    for current_role in ("O", "A", "B", "C"):
        selected = [item for item in cell_records if item["role"] == current_role]
        shell_columns = [[mp.mpf(item["shell_areas"][j]) for item in selected] for j in range(4)]
        angle_values = [mp.mpf(item["boundary_responsibility_angle"]["value"]) for item in selected]
        role_summary[current_role] = {"multiplicity": len(selected), "expected_multiplicity": expected[current_role], "passed_multiplicity": len(selected) == expected[current_role], "total_area": nstr(sum((mp.mpf(item["total_area"]) for item in selected), mp.mpf(0))), "representative_shell_areas": [nstr(values[0]) for values in shell_columns], "mean_shell_areas": [nstr(sum(values, mp.mpf(0)) / len(values)) for values in shell_columns], "shell_spread": [nstr(max(values) - min(values)) for values in shell_columns], "representative_responsibility_angle": nstr(angle_values[0]), "mean_responsibility_angle": nstr(sum(angle_values, mp.mpf(0)) / len(angle_values)), "responsibility_angle_spread": nstr(max(angle_values) - min(angle_values))}
    polar = polar_cross_check(atlas, cell_records, centers, radius_mp, vertices)
    return {"schema": "n25-shell-supercells-v2", "status": "exploration", "input_atlas": {"path": "data/n25_voronoi_atlas.json", "schema": atlas["schema"], "status": atlas["status"], "reported_deterministic_sha256": atlas.get("deterministic_sha256"), "recomputed_deterministic_sha256": recomputed_hash, "recomputed_hash_passed": recomputed_hash == EXPECTED_ATLAS_HASH}, "constants": {"target_radius_squared_exact": qstr(radius_sq[3]), "shell_radii_exact": [qstr(x) for x in radii], "shell_radius_squared_exact": [qstr(x) for x in radius_sq], "shells": ["Omega0=C_t0", "Omega1=C_t1\\C_t0", "Omega2=C_t2\\C_t1", "Omega3=C_R\\C_t2"], "precision_digits": MP_DIGITS}, "method": {"area": "Audited directed Green primitives per edge and radius", "arc_handling": "target boundary arcs use directed short-arc t^2 DeltaTheta/2 and explicit abs(angle)<pi assertion", "verification_status": "high-precision numerical evaluation/display; not an interval"}, "cells": cell_records, "role_summary": role_summary, "boundary_windows": windows, "perfect_matchings": matchings, "double_counting_note": "The 16 BC/CB windows cover the boundary adjacency cycle twice in total; summing all 16 windows double-counts each boundary cell once.", "conservation": conservation, "independent_polar_ray_cross_check": polar, "reproduction": "python src/n25_shell_supercells.py", "limitations": ["Candidate quantity table only.", "Green conservation is a self-consistency check, not a strict interval proof.", "Polar-ray integration is an independent numerical cross-check, not a strict interval proof.", "No local inequality for arbitrary competing clipped Voronoi cells is proved.", "No fixed-radius rigidity, Stage A, Stage B, or global optimality proof is established."], "deterministic_manifest_sha256": ""}


def write_doc(payload, path):
    cons = payload["conservation"]
    polar = payload["independent_polar_ray_cross_check"]
    lines = ["# n=25 shell supercells candidate table", "", "> 状态：**exploration**。这是候选局部量表，不是 fixed-radius certificate、Stage B certificate 或 global optimality proof。", "", "## 复现", "", "```text", payload["reproduction"], "```", "", f"输出 manifest SHA-256：`{payload['deterministic_manifest_sha256']}`。", f"atlas reported SHA-256：`{payload['input_atlas']['reported_deterministic_sha256']}`；按 atlas 规则重算：`{payload['input_atlas']['recomputed_deterministic_sha256']}`；通过：`{payload['input_atlas']['recomputed_hash_passed']}`。", "", "## 方法与语义边界", "", "每个 cell 保存每条有向 edge 的端点、逐半径 Green primitive、line edge 的 quadratic cuts/piece 判定和 target boundary arc 的短弧断言。cell radial range 对 target arc 使用恒定 `R^2`，对内部线段使用 exact-sign 判定投影区间；O cell 的最小平方半径显式为 `0`。", "", "Green 面积与边界角度主值使用 mpmath 90-digit evaluation/display；polar-ray 值及 quadrature error estimate 是 SciPy binary64 数值交叉检查。二者均不是严格区间，也不是证书。", "", "## 关键常数", "", f"壳层半径：`{', '.join(payload['constants']['shell_radii_exact'])}`。目标半径平方：`{payload['constants']['target_radius_squared_exact']}`。", "", "## 角色汇总", "", "| role | multiplicity | representative shell areas | mean shell areas | shell spread | representative angle | mean angle | angle spread |", "|---|---:|---|---|---|---:|---:|---:|"]
    for current_role in ("O", "A", "B", "C"):
        item = payload["role_summary"][current_role]
        lines.append(f"| {current_role} | {item['multiplicity']} | {item['representative_shell_areas']} | {item['mean_shell_areas']} | {item['shell_spread']} | {item['representative_responsibility_angle']} | {item['mean_responsibility_angle']} | {item['responsibility_angle_spread']} |")
    lines += ["", "## Green 守恒自洽检查", "", "| quantity | absolute error | tolerance | passed |", "|---|---:|---:|---:|"]
    for item in cons["shells"] + cons["cumulative"]:
        lines.append(f"| {item.get('shell', item.get('radius'))} | {item['absolute_error']} | {item['tolerance']} | {item['passed']} |")
    total_area = cons["total_area"]
    lines.append(f"| total_area | {total_area['absolute_error']} | {total_area['tolerance']} | {total_area['passed']} |")
    responsibility_angle = cons["responsibility_angle"]
    lines.append(f"| responsibility_angle | {responsibility_angle['absolute_error']} | {responsibility_angle['tolerance']} | {responsibility_angle['passed']} |")
    lines += ["", "## 独立 polar-ray 数值交叉检查", "", f"方法：{polar['method']}；使用 {polar['vertex_polar_breakpoints']} 个 atlas 顶点极角断点。最大绝对误差：`{polar['max_absolute_error']}`；目标 tolerance：`{polar['tolerance']}`；通过：`{polar['passed']}`。检查 cell 为 O、A0、B0、C0 的 t0..t3 cumulative area；记录还包含 scipy quadrature error estimate。", "", "## BC/CB 窗口", "", "生成 `BC_k=B_k union C_k` 与 `CB_k=C_k union B_(k+1 mod 8)` 共 16 个有向窗口；BC 与 CB 各形成一个 8-pair perfect matching；16-window 全求和会双计。", "", "## 限制", "", "本文件及 JSON 仍是 exploration。没有证明任意竞争 cell 的局部不等式、固定半径刚性、Stage A、Stage B 或全局最优性；Green 与 polar-ray 检查都不是严格区间证书。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--json", dest="json_path", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    mp.mp.dps = MP_DIGITS
    atlas = json.loads(args.atlas.read_text(encoding="utf-8"))
    payload = build(atlas)
    payload["deterministic_manifest_sha256"] = canonical_manifest_hash(payload)
    args.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    write_doc(payload, args.doc)
    assert all(item["passed"] for item in payload["conservation"]["shells"] + payload["conservation"]["cumulative"] + [payload["conservation"]["total_area"], payload["conservation"]["responsibility_angle"]])
    assert all(item["joint_responsibility_angle"]["passed"] for item in payload["boundary_windows"])
    assert payload["input_atlas"]["recomputed_hash_passed"]
    assert payload["independent_polar_ray_cross_check"]["passed"]
    print(f"status: {payload['status']}")
    print(f"cells: {len(payload['cells'])}; windows: {len(payload['boundary_windows'])}")
    print(f"atlas recomputed hash: {payload['input_atlas']['recomputed_deterministic_sha256']}")
    print(f"total area error: {payload['conservation']['total_area']['absolute_error']}")
    print(f"polar max error: {payload['independent_polar_ray_cross_check']['max_absolute_error']}")
    print(f"deterministic manifest sha256: {payload['deterministic_manifest_sha256']}")


if __name__ == "__main__":
    main()
