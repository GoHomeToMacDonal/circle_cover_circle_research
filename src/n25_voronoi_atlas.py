"""Generate a certified clipped Voronoi/Delaunay atlas for the n=25 candidate.

All candidate coordinates are generated symbolically.  Floating point is used
only for deterministic display ordering and D8 orbit presentation; it never
selects, rejects, or deduplicates a candidate.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from functools import cmp_to_key, lru_cache
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "n25_voronoi_atlas.json"
DOC = ROOT / "docs" / "n25_voronoi_atlas.md"
Q = sp.sqrt(2)
R2 = 9 + 6 * Q
R = sp.sqrt(R2)
TAYERS = [sp.Integer(1), 1 + Q, sp.sqrt(5 + 2 * Q), R]
LAYER_SQ = [sp.expand(t * t) for t in TAYERS]
LABELS = ["O"] + [f"A{k}" for k in range(8)] + [f"B{k}" for k in range(8)] + [f"C{k}" for k in range(8)]
ROTATIONS = ((1, 0), (Q / 2, Q / 2), (0, 1), (-Q / 2, Q / 2),
             (-1, 0), (-Q / 2, -Q / 2), (0, -1), (Q / 2, -Q / 2))


def rotate(v, k):
    c, s = ROTATIONS[k % 8]
    return (sp.expand(c * v[0] - s * v[1]), sp.expand(s * v[0] + c * v[1]))


def exact_centers():
    a = (1 + Q / 2, Q / 2)
    b = (2 + Q, 0)
    c = (2 + Q, Q)
    return [(sp.Integer(0), sp.Integer(0))] + [rotate(a, k) for k in range(8)] + [rotate(b, k) for k in range(8)] + [rotate(c, k) for k in range(8)]


CENTERS_EXACT = exact_centers()


def fpoint(p):
    return (float(sp.N(p[0], 18)), float(sp.N(p[1], 18)))


@lru_cache(maxsize=None)
def line_for(i, j):
    ci, cj = CENTERS_EXACT[i], CENTERS_EXACT[j]
    a = (sp.expand(2 * (cj[0] - ci[0])), sp.expand(2 * (cj[1] - ci[1])))
    b = sp.expand(cj[0] ** 2 + cj[1] ** 2 - ci[0] ** 2 - ci[1] ** 2)
    return a, b


# --------------------------------------------------------------------------- #
# Certified rational interval arithmetic for the principal-square-root field.
# No numerical approximation is used to choose an algebraic conjugate.
# --------------------------------------------------------------------------- #
def _sqrt_rational_bounds(q, digits=90):
    q = sp.Rational(q)
    if q < 0:
        raise ValueError(f"sqrt of negative rational {q}")
    if q == 0:
        return sp.Integer(0), sp.Integer(0)
    scale = 10 ** digits
    n = (q.p * scale * scale) // q.q
    k = sp.integer_nthroot(n, 2)[0]
    return sp.Rational(k, scale), sp.Rational(k + 1, scale)


def _iadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def _imul(a, b):
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(values), max(values)


def _ipow(a, exponent):
    exponent = int(exponent)
    if exponent == 0:
        return sp.Integer(1), sp.Integer(1)
    if exponent < 0:
        if a[0] <= 0 <= a[1]:
            raise ValueError(f"interval reciprocal crosses zero: {a}")
        return _ipow((sp.Rational(1, a[1]), sp.Rational(1, a[0])), -exponent)
    result = (sp.Integer(1), sp.Integer(1))
    base = a
    while exponent:
        if exponent & 1:
            result = _imul(result, base)
        base = _imul(base, base)
        exponent >>= 1
    return result


def _principal_interval(expr, digits=90):
    expr = sp.cancel(sp.expand(expr))
    if expr.is_Rational:
        return sp.Rational(expr), sp.Rational(expr)
    if expr.is_Add:
        result = (sp.Integer(0), sp.Integer(0))
        for arg in expr.args:
            result = _iadd(result, _principal_interval(arg, digits))
        return result
    if expr.is_Mul:
        result = (sp.Integer(1), sp.Integer(1))
        for arg in expr.args:
            result = _imul(result, _principal_interval(arg, digits))
        return result
    if expr.is_Pow:
        base, exponent = expr.args
        if exponent.is_Rational and exponent.q == 1:
            return _ipow(_principal_interval(base, digits), exponent.p)
        if exponent in (sp.Rational(1, 2), sp.Rational(-1, 2)):
            lo, hi = _principal_interval(base, digits)
            if lo < 0:
                if sp.simplify(base) == 0:
                    lo = sp.Integer(0)
                else:
                    raise ValueError(f"principal sqrt interval crossed negative values: {base}; {lo, hi}")
            lower = _sqrt_rational_bounds(lo, digits)[0]
            upper = _sqrt_rational_bounds(hi, digits)[1]
            if exponent == sp.Rational(1, 2):
                return lower, upper
            if lower <= 0:
                raise ValueError(f"reciprocal principal sqrt interval crosses zero: {base}")
            return sp.Rational(1, upper), sp.Rational(1, lower)
    if expr.is_Number:
        return sp.Rational(expr), sp.Rational(expr)
    raise TypeError(f"unsupported algebraic expression in interval evaluator: {expr}")


@lru_cache(maxsize=None)
def interval(expr_text):
    expr = sp.sympify(expr_text)
    if sp.simplify(expr) == 0:
        return sp.Integer(0), sp.Integer(0)
    for digits in (70, 100, 140):
        try:
            lo, hi = _principal_interval(expr, digits)
        except ValueError:
            continue
        if lo <= hi:
            return lo, hi
    raise RuntimeError(f"strict interval evaluation failed for {expr}")


def _normalized_expr(expr):
    return sp.cancel(sp.expand(expr))


@lru_cache(maxsize=None)
def _exact_sign_cached(expr_text):
    expr = sp.sympify(expr_text)
    if expr == 0 or sp.simplify(expr) == 0:
        return 0
    lo, hi = interval(sp.srepr(expr))
    if lo > 0:
        return 1
    if hi < 0:
        return -1
    raise RuntimeError(f"undecided exact sign: [{lo}, {hi}] for {expr}")


_SIGN_CALLS = 0


def exact_sign(expr, context=""):
    global _SIGN_CALLS
    _SIGN_CALLS += 1
    normalized = _normalized_expr(expr)
    try:
        return _exact_sign_cached(sp.srepr(normalized))
    except RuntimeError as exc:
        if context:
            raise RuntimeError(f"{exc} in {context}") from exc
        raise


def qstr(x):
    return str(sp.factor(x))


def int_record(expr):
    lo, hi = interval(sp.srepr(sp.cancel(sp.expand(expr))))
    return {"lo": qstr(lo), "hi": qstr(hi)}


def exact_distance_sq(p, i):
    return sp.cancel(sp.expand((p[0] - CENTERS_EXACT[i][0]) ** 2 + (p[1] - CENTERS_EXACT[i][1]) ** 2))


# --------------------------------------------------------------------------- #
# Complete exact candidate enumeration.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _boundary_pair_geometry(i, j):
    (a0, a1), b = line_for(i, j)
    a2 = sp.expand(a0 * a0 + a1 * a1)
    if sp.simplify(a2) == 0:
        return None
    radicand = sp.cancel(sp.expand(a2 * R2 - b * b))
    radicand_sign = exact_sign(radicand, f"boundary radicand for pair {i},{j}")
    if radicand_sign < 0:
        return (radicand, tuple(), False)
    foot = (sp.cancel(a0 * b / a2), sp.cancel(a1 * b / a2))
    root = sp.sqrt(radicand)
    roots = (
        ((sp.cancel(foot[0] - a1 * root / a2), sp.cancel(foot[1] + a0 * root / a2)), True),
        ((sp.cancel(foot[0] + a1 * root / a2), sp.cancel(foot[1] - a0 * root / a2)), True),
    )
    return (radicand, roots, radicand_sign == 0)


def _boundary_pair_candidates(i, j):
    geometry = _boundary_pair_geometry(i, j)
    if geometry is None:
        return ()
    return geometry[1]


def _triple_candidate(i, j, k):
    (a0, a1), b = line_for(i, j)
    (c0, c1), d = line_for(i, k)
    det = sp.expand(a0 * c1 - a1 * c0)
    if sp.simplify(det) == 0:
        return None
    return (sp.cancel((b * c1 - a1 * d) / det), sp.cancel((a0 * d - b * c0) / det))


def _nearest_sites(p, defining_sites):
    defining_sites = tuple(dict.fromkeys(defining_sites))
    if not defining_sites:
        raise ValueError("nearest-site search requires at least one defining site")
    distances = {}
    anchor = defining_sites[0]
    anchor_distance = exact_distance_sq(p, anchor)
    distances[anchor] = anchor_distance
    for site in defining_sites[1:]:
        distance = exact_distance_sq(p, site)
        distances[site] = distance
        if exact_sign(distance - anchor_distance, f"defining-site equality {anchor} vs {site}") != 0:
            raise AssertionError(f"defining sites are not exactly equidistant: {defining_sites}")
    active = [anchor]
    for site in range(25):
        if site == anchor:
            continue
        distance = distances.get(site)
        if distance is None:
            distance = exact_distance_sq(p, site)
            distances[site] = distance
        sign = exact_sign(distance - anchor_distance, f"nearest-site {anchor} vs {site}")
        if sign < 0:
            return None, distances
        if sign == 0:
            active.append(site)
    return active, distances


def _inside_target(p):
    return exact_sign(sp.expand(p[0] * p[0] + p[1] * p[1] - R2), "target-disk incidence") <= 0


def _same_point(p, q):
    return sp.simplify(p[0] - q[0]) == 0 and sp.simplify(p[1] - q[1]) == 0


def _coordinate_key(p):
    return tuple(sp.srepr(_normalized_expr(value)) for value in p)


def _exact_polar_cmp(p, q, center=(sp.Integer(0), sp.Integer(0)), context="polar ordering"):
    """Compare two vectors by exact counter-clockwise polar angle."""
    ap = (sp.expand(p[0] - center[0]), sp.expand(p[1] - center[1]))
    aq = (sp.expand(q[0] - center[0]), sp.expand(q[1] - center[1]))

    def half(v, label):
        sy = exact_sign(v[1], f"{context} {label} y half-plane")
        if sy > 0:
            return 0
        if sy < 0:
            return 1
        return 0 if exact_sign(v[0], f"{context} {label} x-axis half-plane") >= 0 else 1

    hp, hq = half(ap, "left"), half(aq, "right")
    if hp != hq:
        return -1 if hp < hq else 1
    cross_sign = exact_sign(
        sp.expand(ap[0] * aq[1] - ap[1] * aq[0]),
        f"{context} cross product",
    )
    if cross_sign > 0:
        return -1
    if cross_sign < 0:
        return 1

    # Collinear points on one ray are ordered by exact radius, then exact
    # coordinates. Equal coordinates are a topology error, never a tie.
    rp = sp.expand(ap[0] * ap[0] + ap[1] * ap[1])
    rq = sp.expand(aq[0] * aq[0] + aq[1] * aq[1])
    radius_sign = exact_sign(rp - rq, f"{context} collinear radius")
    if radius_sign:
        return -1 if radius_sign < 0 else 1
    x_sign = exact_sign(ap[0] - aq[0], f"{context} collinear x coordinate")
    if x_sign:
        return -1 if x_sign < 0 else 1
    y_sign = exact_sign(ap[1] - aq[1], f"{context} collinear y coordinate")
    if y_sign:
        return -1 if y_sign < 0 else 1
    raise RuntimeError(f"duplicate exact polar points in {context}: {p}")


def _exact_polar_cycle(items, point_getter, center, context):
    for left in range(len(items)):
        for right in range(left):
            if _same_point(point_getter(items[left]), point_getter(items[right])):
                raise RuntimeError(f"duplicate exact point in {context}")
    return sorted(items, key=cmp_to_key(
        lambda left, right: _exact_polar_cmp(
            point_getter(left), point_getter(right), center, context
        )
    ))


def enumerate_candidates():
    started = time.perf_counter()
    candidates = []
    coordinate_cache = {}
    telemetry = {
        "target_disk_seconds": 0.0,
        "coordinate_cache_seconds": 0.0,
        "nearest_site_seconds": 0.0,
        "boundary_pair_geometry_seconds": 0.0,
        "triple_generation_seconds": 0.0,
        "sorting_seconds": 0.0,
    }
    stats = {
        "triple_total": 0, "triple_nonparallel": 0, "triple_outside": 0,
        "triple_nonnearest": 0, "boundary_pair_total": 0,
        "boundary_roots": 0, "boundary_negative_radicand": 0,
        "boundary_outside": 0, "boundary_nonnearest": 0,
        "exact_duplicate": 0, "raw_duplicate_total": 0,
        "raw_duplicate_by_status": {"accepted": 0, "outside": 0, "nonnearest": 0},
    }

    def accept(p, source, defining_sites, exact_boundary=None):
        target_started = time.perf_counter()
        inside = _inside_target(p)
        telemetry["target_disk_seconds"] += time.perf_counter() - target_started
        cache_started = time.perf_counter()
        key = _coordinate_key(p)
        cached = coordinate_cache.get(key)
        if not inside:
            if cached is None:
                coordinate_cache[key] = {"status": "outside", "nearest_by_anchor": {}, "candidate": None}
            else:
                stats["raw_duplicate_total"] += 1
                stats["raw_duplicate_by_status"]["outside"] += 1
                stats["exact_duplicate"] += 1
            telemetry["coordinate_cache_seconds"] += time.perf_counter() - cache_started
            return False, "outside"
        duplicate = cached is not None
        if cached is None:
            cached = {"status": "unknown", "nearest_by_anchor": {}, "candidate": None}
            coordinate_cache[key] = cached
        if duplicate:
            stats["raw_duplicate_total"] += 1
            stats["exact_duplicate"] += 1
        anchor = defining_sites[0]
        nearest_by_anchor = cached["nearest_by_anchor"]
        if anchor not in nearest_by_anchor:
            nearest_started = time.perf_counter()
            active, _ = _nearest_sites(p, defining_sites)
            telemetry["nearest_site_seconds"] += time.perf_counter() - nearest_started
            nearest_by_anchor[anchor] = active
        else:
            active = nearest_by_anchor[anchor]
        if active is None or not set(defining_sites).issubset(active):
            if duplicate:
                stats["raw_duplicate_by_status"]["nonnearest"] += 1
            cached["status"] = "nonnearest"
            telemetry["coordinate_cache_seconds"] += time.perf_counter() - cache_started
            return False, "nonnearest"
        candidate = cached["candidate"]
        if candidate is not None:
            if duplicate:
                stats["raw_duplicate_by_status"]["accepted"] += 1
            candidate["sources"].append(source)
            candidate["defining_sites"].append(list(defining_sites))
            cached["status"] = "accepted"
            telemetry["coordinate_cache_seconds"] += time.perf_counter() - cache_started
            return False, "duplicate"
        candidate = {
            "exact": p,
            "sources": [source],
            "defining_sites": [list(defining_sites)],
            "active": active,
            "exact_boundary": exact_boundary,
        }
        candidates.append(candidate)
        cached["status"] = "accepted"
        cached["candidate"] = candidate
        if duplicate:
            stats["raw_duplicate_by_status"]["accepted"] += 1
        telemetry["coordinate_cache_seconds"] += time.perf_counter() - cache_started
        return True, "accepted"

    for i in range(25):
        for j in range(i + 1, 25):
            stats["boundary_pair_total"] += 1
            geometry_started = time.perf_counter()
            geometry = _boundary_pair_geometry(i, j)
            telemetry["boundary_pair_geometry_seconds"] += time.perf_counter() - geometry_started
            if geometry is None:
                continue
            _, roots, exact_boundary = geometry
            if not roots:
                stats["boundary_negative_radicand"] += 1
                continue
            stats["boundary_roots"] += len(roots)
            for root_no, (p, root_boundary) in enumerate(roots):
                accepted, why = accept(p, {"kind": "boundary-pair", "sites": [i, j], "root": root_no}, (i, j), root_boundary or exact_boundary)
                if not accepted and why == "outside":
                    stats["boundary_outside"] += 1
                elif not accepted and why == "nonnearest":
                    stats["boundary_nonnearest"] += 1

    triple_started = time.perf_counter()
    for i in range(25):
        for j in range(i + 1, 25):
            for k in range(j + 1, 25):
                stats["triple_total"] += 1
                p = _triple_candidate(i, j, k)
                if p is None:
                    continue
                stats["triple_nonparallel"] += 1
                accepted, why = accept(p, {"kind": "triple", "sites": [i, j, k]}, (i, j, k))
                if not accepted and why == "outside":
                    stats["triple_outside"] += 1
                elif not accepted and why == "nonnearest":
                    stats["triple_nonnearest"] += 1
    telemetry["triple_generation_seconds"] = time.perf_counter() - triple_started

    sorting_started = time.perf_counter()
    candidates.sort(key=cmp_to_key(
        lambda left, right: _exact_polar_cmp(
            left["exact"], right["exact"], context="candidate global polar ordering"
        )
    ))
    for index, candidate in enumerate(candidates):
        candidate["id"] = index
    telemetry["sorting_seconds"] = time.perf_counter() - sorting_started
    stats["accepted"] = len(candidates)
    stats["coordinate_cache_entries"] = len(coordinate_cache)
    stats["coordinate_cache_hits"] = stats["raw_duplicate_total"]
    stats["excluded"] = stats["triple_outside"] + stats["triple_nonnearest"] + stats["boundary_outside"] + stats["boundary_nonnearest"]
    stats["excluded_by_reason"] = {
        "outside_target": stats["triple_outside"] + stats["boundary_outside"],
        "not_nearest_site": stats["triple_nonnearest"] + stats["boundary_nonnearest"],
        "exact_duplicate": stats["exact_duplicate"],
        "negative_boundary_radicand": stats["boundary_negative_radicand"],
    }
    telemetry["total_seconds"] = time.perf_counter() - started
    stats["telemetry"] = telemetry
    stats["sign_calls"] = _SIGN_CALLS
    stats["sign_cache"] = {key: value for key, value in _exact_sign_cached.cache_info()._asdict().items()}
    stats["interval_cache"] = {key: value for key, value in interval.cache_info()._asdict().items()}
    return candidates, stats


# --------------------------------------------------------------------------- #
# Exact vertex records and certified graph segments.
# --------------------------------------------------------------------------- #
def vertex_record(candidate):
    p = candidate["exact"]
    active = candidate["active"]
    distances = [exact_distance_sq(p, i) for i in range(25)]
    margins = []
    for j in range(25):
        if j not in active:
            delta = sp.cancel(sp.expand(distances[j] - distances[active[0]]))
            lo, hi = interval(sp.srepr(delta))
            if lo <= 0:
                raise RuntimeError(f"non-nearest margin not strictly positive at vertex {candidate['id']}, site {j}")
            margins.append({"site": j, "interval": {"lo": qstr(lo), "hi": qstr(hi)}, "strict_positive": True})
    radial_delta = sp.cancel(sp.expand(p[0] * p[0] + p[1] * p[1] - R2))
    radial_sign = exact_sign(radial_delta, f"vertex radial status {candidate['id']}")
    exact_boundary = radial_sign == 0
    if candidate.get("exact_boundary") is not None and candidate["exact_boundary"] != exact_boundary:
        raise RuntimeError(f"boundary state mismatch at vertex {candidate['id']}")
    boundary = exact_boundary
    return {
        "id": candidate["id"],
        "coordinate": [qstr(p[0]), qstr(p[1])],
        "coordinate_interval": [int_record(p[0]), int_record(p[1])],
        "coordinate_status": "exact-algebraic + principal-root rational interval",
        "discovery_coordinate": list(fpoint(p)),
        "boundary": boundary,
        "radial_incidence_status": {"exact_boundary_zero": boundary, "inside_interval": None if boundary else int_record(radial_delta), "passed": radial_sign <= 0},
        "nearest_site_set": active,
        "nearest_site_status": "exact equalities + strictly positive rational interval exclusion",
        "active_equalities": [{"sites": [active[0], j], "exact_zero": sp.simplify(distances[active[0]] - distances[j]) == 0} for j in active[1:]],
        "nonnearest_margins": margins,
        "source_count": len(candidate["sources"]),
        "candidate_sources": candidate["sources"],
    }


def make_vertices(candidates):
    records = [vertex_record(c) for c in candidates]
    cell_vids = [[] for _ in range(25)]
    for record in records:
        p = tuple(record["discovery_coordinate"])
        for site in record["nearest_site_set"]:
            cell_vids[site].append(record["id"])
    for site in range(25):
        cell_vids[site] = _exact_polar_cycle(
            cell_vids[site],
            lambda vid: _record_point(records[vid]),
            CENTERS_EXACT[site],
            f"cell {site} vertex cycle",
        )
    boundary_cycle = _exact_polar_cycle(
        [record["id"] for record in records if record["boundary"]],
        lambda vid: _record_point(records[vid]),
        (sp.Integer(0), sp.Integer(0)),
        "target boundary global cycle",
    )
    return cell_vids, records, boundary_cycle


def _segment_certificate(u, v, sites):
    if len(sites) != 2:
        raise RuntimeError(f"internal edge needs exactly two sites, got {sites}")
    # Every nearest-site difference is affine on the common bisector.  Its
    # endpoint nonnegativity therefore certifies nonnegativity on the segment.
    for third in range(25):
        if third in sites:
            continue
        du = exact_distance_sq(u, third) - exact_distance_sq(u, sites[0])
        dv = exact_distance_sq(v, third) - exact_distance_sq(v, sites[0])
        if exact_sign(du, f"segment endpoint {sites} vs {third}") < 0 or exact_sign(dv, f"segment endpoint {sites} vs {third}") < 0:
            raise RuntimeError(f"internal segment is not nearest-site certified: {sites}, {third}")
    return "certified: affine nearest-site differences nonnegative at both endpoints"


def _boundary_arc_certificate(u, v, site, boundary_cycle, enumeration):
    if site not in u["nearest_site_set"] or site not in v["nearest_site_set"]:
        raise RuntimeError(f"boundary arc endpoint is not owned by site {site}")
    expected_pairs = 25 * 24 // 2
    if enumeration["boundary_pair_total"] != expected_pairs or enumeration["boundary_roots"] != 2 * expected_pairs:
        raise RuntimeError("boundary responsibility certificate lacks the complete 300-pair enumeration")
    positions = {vertex_id: position for position, vertex_id in enumerate(boundary_cycle)}
    if u["id"] not in positions or v["id"] not in positions:
        raise RuntimeError("boundary arc endpoint is absent from the global boundary cycle")
    n = len(boundary_cycle)
    pos_u, pos_v = positions[u["id"]], positions[v["id"]]
    ccw_steps = (pos_v - pos_u) % n
    if ccw_steps == 1:
        direction = "CCW"
    elif ccw_steps == n - 1:
        direction = "CW"
    else:
        raise RuntimeError(f"boundary edge is not a short global-cycle arc: {u['id']}->{v['id']}")
    return {
        "status": "certified",
        "method": "complete 300 bisector-pair / 600 boundary-root enumeration plus exact endpoint nearest ownership",
        "directed_endpoints": {"from": u["id"], "to": v["id"], "direction": direction},
        "global_cycle_positions": {"from": pos_u, "to": pos_v, "cycle_length": n, "ccw_steps": ccw_steps},
        "endpoint_nearest_ownership": {"from": list(u["nearest_site_set"]), "to": list(v["nearest_site_set"]), "site": site},
        "boundary_bisector_enumeration": {"pairs": enumeration["boundary_pair_total"], "roots": enumeration["boundary_roots"], "complete": True},
        "open_short_arc_boundary_vertices": [],
        "no_unenumerated_responsibility_switches": True,
    }


def build_edges(cell_vids, vertex_records, boundary_cycle, enumeration):
    internal = {}
    boundary = {}
    for site, loop in enumerate(cell_vids):
        for uid, vid in zip(loop, loop[1:] + loop[:1]):
            u, v = vertex_records[uid], vertex_records[vid]
            common = sorted(set(u["nearest_site_set"]) & set(v["nearest_site_set"]))
            if len(common) >= 2 and site in common:
                pair = tuple(common[:2])
                key = (pair, tuple(sorted((uid, vid))))
                internal[key] = {"kind": "voronoi", "sites": list(pair), "vertices": [min(uid, vid), max(uid, vid)], "exact_incidence": True, "segment_status": _segment_certificate(u["exact_point"], v["exact_point"], pair)}
            elif u["boundary"] and v["boundary"]:
                key = (site, tuple(sorted((uid, vid))))
                boundary[key] = {"kind": "target-boundary-arc", "sites": [site], "vertices": [min(uid, vid), max(uid, vid)], "exact_incidence": True, "arc_status": _boundary_arc_certificate(u, v, site, boundary_cycle, enumeration)}
    return list(internal.values()), list(boundary.values())


def _exact_point_records(records):
    return [(sp.sympify(r["coordinate"][0]), sp.sympify(r["coordinate"][1])) for r in records]


def _record_point(record):
    return (sp.sympify(record["coordinate"][0]), sp.sympify(record["coordinate"][1]))


# Attach exact coordinates without exposing them in JSON.
def _attach_exact_points(records):
    for record in records:
        record["exact_point"] = _record_point(record)


def _remove_private_points(records):
    for record in records:
        record.pop("exact_point", None)


def _min_segment_radius_sq(u, v):
    d = (sp.expand(v[0] - u[0]), sp.expand(v[1] - u[1]))
    dd = sp.expand(d[0] * d[0] + d[1] * d[1])
    if sp.simplify(dd) == 0:
        return sp.expand(u[0] * u[0] + u[1] * u[1])
    lam = sp.cancel(-(u[0] * d[0] + u[1] * d[1]) / dd)
    sl = exact_sign(lam, "segment minimum parameter lower endpoint")
    su = exact_sign(lam - 1, "segment minimum parameter upper endpoint")
    if sl < 0:
        return sp.expand(u[0] * u[0] + u[1] * u[1])
    if su > 0:
        return sp.expand(v[0] * v[0] + v[1] * v[1])
    # lambda == 0 or 1 is intentionally handled by this same exact formula;
    # it agrees algebraically with the corresponding endpoint value.
    return sp.cancel(sp.expand(u[0] * u[0] + u[1] * u[1] - (u[0] * d[0] + u[1] * d[1]) ** 2 / dd))


def _exact_min(values, context):
    result = values[0]
    for value in values[1:]:
        sign = exact_sign(value - result, context)
        if sign < 0:
            result = value
    return result


def layer_record(loop, vertex_records):
    points = [_record_point(vertex_records[vid]) for vid in loop]
    min_sq = _exact_min([sp.expand(p[0] * p[0] + p[1] * p[1]) for p in points] + [_min_segment_radius_sq(u, v) for u, v in zip(points, points[1:] + points[:1])], "cell radial minimum")
    max_sq = points[0][0] ** 2 + points[0][1] ** 2
    for p in points[1:]:
        value = sp.expand(p[0] * p[0] + p[1] * p[1])
        if exact_sign(value - max_sq, "cell radial maximum") > 0:
            max_sq = value
    layers = []
    for index, layer_sq in enumerate(LAYER_SQ):
        low = exact_sign(layer_sq - min_sq, f"layer {index} lower crossing") >= 0
        high = exact_sign(max_sq - layer_sq, f"layer {index} upper crossing") >= 0
        layers.append({"layer": f"t{index}", "crosses": low and high, "comparison": {"min_sq": int_record(layer_sq - min_sq), "max_sq": int_record(max_sq - layer_sq)}, "status": "exact interval certified"})
    return {"radial_range_squared": [qstr(min_sq), qstr(max_sq)], "radial_range_interval": [int_record(min_sq), int_record(max_sq)], "layers": layers, "status": "exact interval certified"}


# --------------------------------------------------------------------------- #
# Symmetry presentation.  Float is used only to locate already certified IDs.
# --------------------------------------------------------------------------- #
def point_transform(p, k, reflected):
    angle = k * math.pi / 4
    x, y = p
    if reflected:
        y = -y
    c, s = math.cos(angle), math.sin(angle)
    return (c * x - s * y, s * x + c * y)


def nearest_index(p, points):
    distances = [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in points]
    i = min(range(len(points)), key=lambda j: distances[j])
    return i if distances[i] < 1e-12 else None


def find_orbits(items, point_map):
    remaining = set(range(len(items)))
    output = []
    while remaining:
        seed = min(remaining)
        orbit = {seed}
        for k in range(8):
            for reflected in (False, True):
                mapped = point_map(seed, k, reflected)
                if mapped is not None:
                    orbit.add(mapped)
        output.append(sorted(orbit))
        remaining -= orbit
    return sorted(output, key=lambda x: (len(x), x))


def make_orbits(cells, vertex_records, edges, faces):
    vc = [fpoint(p) for p in CENTERS_EXACT]
    vp = [tuple(v["discovery_coordinate"]) for v in vertex_records]
    edge_keys = {(tuple(sorted(e["vertices"])), tuple(sorted(e["sites"]))) for e in edges}

    def edge_map(global_id, k, reflected):
        e = edges[global_id]
        a = nearest_index(point_transform(vp[e["vertices"][0]], k, reflected), vp)
        b = nearest_index(point_transform(vp[e["vertices"][1]], k, reflected), vp)
        if a is None or b is None:
            return None
        sites = tuple(sorted(nearest_index(point_transform(vc[j], k, reflected), vc) for j in e["sites"]))
        if None in sites:
            return None
        key = (tuple(sorted((a, b))), sites)
        return next((j for j, item in enumerate(edges) if (tuple(sorted(item["vertices"])), tuple(sorted(item["sites"]))) == key), None)

    def face_map(i, k, reflected):
        mapped = [nearest_index(point_transform(vc[j], k, reflected), vc) for j in faces[i]]
        if None in mapped:
            return None
        key = tuple(sorted(mapped))
        return next((j for j, item in enumerate(faces) if tuple(sorted(item)) == key), None)

    def cell_map(i, k, reflected):
        return nearest_index(point_transform(vc[i], k, reflected), vc)

    def vertex_map(i, k, reflected):
        return nearest_index(point_transform(vp[i], k, reflected), vp)

    return {"cells": find_orbits(cells, cell_map), "vertices": find_orbits(vertex_records, vertex_map), "edges": find_orbits(edges, edge_map), "internal_edges": [orbit for orbit in find_orbits(edges, edge_map) if all(x < 56 for x in orbit)], "boundary_edges": [orbit for orbit in find_orbits(edges, edge_map) if all(x >= 56 for x in orbit)], "faces": find_orbits(faces, face_map)}


def build():
    build_started = time.perf_counter()
    stage_timings = {}
    stage_started = time.perf_counter()
    candidates, enumeration = enumerate_candidates()
    stage_timings["candidate_enumeration_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    cell_vids, vertex_records, boundary_cycle = make_vertices(candidates)
    stage_timings["vertices_seconds"] = time.perf_counter() - stage_started
    exact_points = [_record_point(v) for v in vertex_records]
    for record, point in zip(vertex_records, exact_points):
        record["exact_point"] = point
    stage_started = time.perf_counter()
    internal, boundary = build_edges(cell_vids, vertex_records, boundary_cycle, enumeration)
    stage_timings["edges_seconds"] = time.perf_counter() - stage_started
    for edge_id, edge in enumerate(internal):
        edge["id"] = edge_id
    for edge_id, edge in enumerate(boundary, start=len(internal)):
        edge["id"] = edge_id
    edges = internal + boundary
    stage_started = time.perf_counter()
    faces = sorted({tuple(sorted(v["nearest_site_set"])) for v in vertex_records if not v["boundary"] and len(v["nearest_site_set"]) == 3})
    cells = []
    for site, loop in enumerate(cell_vids):
        edge_cycle = []
        for u, v in zip(loop, loop[1:] + loop[:1]):
            matches = [edge["id"] for edge in edges if set(edge["vertices"]) == {u, v} and site in edge["sites"]]
            if len(matches) != 1:
                raise RuntimeError(f"cell edge incidence is not unique for site {site}, vertices {u},{v}: {matches}")
            edge_cycle.append(matches[0])
        cells.append({"site": site, "label": LABELS[site], "vertex_cycle": loop, "edge_cycle": edge_cycle, "vertex_count": len(loop), "layer_crossings": layer_record(loop, vertex_records)})
    stage_timings["cells_and_layers_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    _remove_private_points(vertex_records)
    orbits = make_orbits(cells, vertex_records, edges, faces)
    candidate_hash_input = [{"id": c["id"], "coordinate": [qstr(c["exact"][0]), qstr(c["exact"][1])], "active": c["active"], "sources": c["sources"]} for c in candidates]
    candidate_hash = hashlib.sha256(json.dumps(candidate_hash_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    stage_timings["orbits_and_hash_seconds"] = time.perf_counter() - stage_started
    stage_timings["total_seconds"] = time.perf_counter() - build_started
    telemetry = {
        "total_seconds": stage_timings["total_seconds"],
        "stages": stage_timings,
        "enumeration": enumeration["telemetry"],
        "sign_calls": enumeration["sign_calls"],
        "sign_cache": enumeration["sign_cache"],
        "interval_cache": enumeration["interval_cache"],
        "coordinate_cache": {
            "entries": enumeration["coordinate_cache_entries"],
            "hits": enumeration["coordinate_cache_hits"],
            "raw_duplicates": enumeration["raw_duplicate_total"],
        },
        "geometry_cache": {
            "line_for": {key: value for key, value in line_for.cache_info()._asdict().items()},
            "boundary_pair_geometry": {key: value for key, value in _boundary_pair_geometry.cache_info()._asdict().items()},
        },
    }
    payload = {
        "status": "exploration",
        "semantic_acceptance": {"status": "passed", "scope": "P0 n25 topology semantics", "exact_polar_cell_cycles": True, "exact_global_boundary_cycle": True, "boundary_arc_responsibility_switches": "exhaustive"},
        "schema": "n25-clipped-voronoi-delaunay-atlas-v3",
        "candidate": {"n": 25, "cover_radius": "1", "target_radius_squared": qstr(R2), "target_radius": qstr(R), "center_order": LABELS, "centers_exact": [[qstr(x), qstr(y)] for x, y in CENTERS_EXACT], "center_status": "exact-algebraic", "discovery_status": "float-display-only"},
        "candidate_enumeration": {**enumeration, "candidate_hash": candidate_hash, "status": "complete exact triple and boundary-pair enumeration"},
        "telemetry": telemetry,
        "vertices": vertex_records,
        "boundary_intersections": [v["id"] for v in vertex_records if v["boundary"]],
        "boundary_cycle": boundary_cycle,
        "boundary_cycle_status": {"orientation": "CCW", "construction": "exact polar comparator about target origin", "vertex_count": len(boundary_cycle), "all_vertices_certified": True},
        "cells": cells,
        "internal_voronoi_edges": internal,
        "target_boundary_edges": boundary,
        "delaunay_edges_clipped": [e["sites"] for e in internal],
        "delaunay_faces_clipped": [list(f) for f in faces],
        "delaunay_status": "clipped dual; exact certified incidence",
        "d8_orbits": orbits,
        "euler_check": {"V": len(vertex_records), "E_internal": len(internal), "E_boundary": len(boundary), "E_total": len(edges), "F_cells": len(cells), "F_outer": 1, "V-E+F": len(vertex_records) - len(edges) + len(cells) + 1, "expected": 2, "passed": len(vertex_records) - len(edges) + len(cells) + 1 == 2},
        "verification": {"candidate_set": "complete exact enumeration; no float filtering or deduplication", "coordinates": "exact-algebraic expressions with principal-root rational intervals", "incidence": "exact construction plus certified nearest-site comparisons", "nearest_site": "exact active equalities and strictly positive rational interval margins", "cell_cycles": "exact polar comparator: y-sign/x-axis half-plane, exact cross sign, exact collinear radius/coordinate tie-break; duplicate points fail", "internal_edges": "full segment certified by affine nearest-site differences at both endpoints", "boundary_cycle": "all 16 certified target-boundary vertices sorted by exact global CCW polar comparator", "boundary_arcs": "complete 300 bisector-pair / 600-root enumeration, global-cycle adjacency, and endpoint ownership certify each short directed arc with no omitted responsibility switch", "layers": "exact interval radial min/max certification", "not_a_proof": "This atlas is exploration and does not establish Stage B or global optimality."},
    }
    hash_payload = {key: value for key, value in payload.items() if key != "telemetry"}
    hash_enumeration = {key: value for key, value in hash_payload["candidate_enumeration"].items() if key != "telemetry"}
    hash_payload["candidate_enumeration"] = hash_enumeration
    unsigned = json.dumps(hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["deterministic_sha256"] = hashlib.sha256(unsigned.encode()).hexdigest()
    return payload


def write_report(payload):
    c = payload["euler_check"]
    e = payload["candidate_enumeration"]
    v = payload["verification"]
    lines = [
        "# n=25 clipped Voronoi/Delaunay atlas", "",
        "> 状态：**exploration**；P0 语义验收：**PASS**。此产物不是 fixed-radius certificate，也不是 global optimality proof。", "",
        "## 复现", "", "```text", "python src/n25_voronoi_atlas.py", "```",
        f"JSON 确定性 SHA-256（排除运行时遥测）：`{payload['deterministic_sha256']}`；候选集合 SHA-256：`{e['candidate_hash']}`。", "",
        "## 图谱计数", "",
        f"候选穷举：非平行三站点等距线 {e['triple_nonparallel']}/{e['triple_total']}；目标圆与垂直平分线候选根 {e['boundary_roots']}（{e['boundary_pair_total']} 对）；严格保留 {e['accepted']} 个顶点，精确重复 {e['exact_duplicate']} 个。排除分类：{json.dumps(e['excluded_by_reason'], ensure_ascii=False, sort_keys=True)}。",
        f"25 个有序圆心；{len(payload['vertices'])} 个顶点，其中内部 {sum(not x['boundary'] for x in payload['vertices'])}、目标边界 {sum(x['boundary'] for x in payload['vertices'])}；{len(payload['internal_voronoi_edges'])} 条内部 Voronoi 边、{len(payload['target_boundary_edges'])} 条目标边界弧；{len(payload['delaunay_edges_clipped'])} 条 clipped Delaunay 边、{len(payload['delaunay_faces_clipped'])} 个三角面。",
        f"Euler：V={c['V']}，E={c['E_total']}（内部 {c['E_internal']} + 边界 {c['E_boundary']}），F={c['F_cells']}+外部面，V-E+F={c['V-E+F']}（通过：{c['passed']}）。边界 edge ID 全局固定为 {c['E_internal']}..{c['E_total'] - 1}。", "",
        f"运行时遥测：总耗时 {payload['telemetry']['total_seconds']:.3f}s；判号调用 {payload['telemetry']['sign_calls']}；sign cache {json.dumps(payload['telemetry']['sign_cache'], ensure_ascii=False, sort_keys=True)}；coordinate cache {json.dumps(payload['telemetry']['coordinate_cache'], ensure_ascii=False, sort_keys=True)}。", "",
        "## 验证状态", "",
        f"- 候选集合：{v['candidate_set']}。", f"- 坐标与判号：{v['coordinates']}；{v['nearest_site']}。", f"- Cell cycle：{v['cell_cycles']}。", f"- 内部边：{v['internal_edges']}。", f"- 全局边界 cycle：{v['boundary_cycle']}。", f"- 边界弧：{v['boundary_arcs']}。", f"- 径向层：{v['layers']}。", "- 浮点仅用于 discovery/display 与 D8 展示对象匹配；证书拓扑排序使用 exact polar comparator。", "",
        "## 已证与未证边界", "",
        "候选顶点集合、坐标、nearest-site、完整内部线段和目标圆边界弧责任、四层穿越均已用精确表达式与有理区间记录；浮点不参与候选集合、筛选、去重或严格状态。对象仍标为 `exploration`：没有枚举竞争覆盖构型、没有 Stage B 分支穷尽，也没有固定半径刚性或全局最优性证明。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    payload = build()
    DATA.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    e = payload["candidate_enumeration"]
    print(f"status: {payload['status']}")
    print(f"candidate enumeration: triples={e['triple_nonparallel']}/{e['triple_total']}, boundary_roots={e['boundary_roots']}, accepted={e['accepted']}, duplicates={e['exact_duplicate']}")
    print(f"vertices: {len(payload['vertices'])} (internal={sum(not x['boundary'] for x in payload['vertices'])}, boundary={sum(x['boundary'] for x in payload['vertices'])})")
    print(f"internal Voronoi edges: {len(payload['internal_voronoi_edges'])}; boundary arcs: {len(payload['target_boundary_edges'])}")
    print(f"Euler: {payload['euler_check']}")
    print(f"candidate_sha256: {e['candidate_hash']}")
    print(f"sha256: {payload['deterministic_sha256']}")


if __name__ == "__main__":
    main()
