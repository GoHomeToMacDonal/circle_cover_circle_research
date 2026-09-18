from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
SHELL_PATH = ROOT / "data" / "n25_shell_supercells.json"
P1_PATH = ROOT / "data" / "n25_p1_smoke.json"
ATLAS_PATH = ROOT / "data" / "n25_voronoi_atlas.json"
OUT_PATH = ROOT / "data" / "n25_p2_potential_search.json"
DOC_PATH = ROOT / "docs" / "n25_p2_potential_search.md"

EXPECTED_SHELL_HASH = "d9387b381aa1e20d31df7873792e041c88de55f61585196777efabaa53b75d49"
EXPECTED_P1_HASH = "06fefe31636bfca216834aceeb3121a6962a6a05950ced62252cd2520108e433"
EXPECTED_ATLAS_HASH = "cda22f3719de1e20eb100ae011667fbe25952137b82be77c74e8b5726fc92257"
GRID = 8192
BOUNDARY_GRID = 8192
SERVICE_TOL = 1e-10
SERVICE_GUARD = 1e-5
DEFICIT_THRESHOLD = 1e-6
MIN_STRICT_SAMPLES = 25
SCAN_POINTS = 4096
REFINE_POINTS = 1025
REFINE_GRID = 4 * GRID
PERTURB_SIGMAS = (1e-4, 1e-3, 1e-2, 5e-2)


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def shell_manifest_hash(payload: dict) -> str:
    material = copy.deepcopy(payload)
    material.pop("telemetry", None)
    material.pop("deterministic_manifest_sha256", None)
    return canonical_hash(material)


def p1_manifest_hash(payload: dict) -> str:
    material = copy.deepcopy(payload)
    material.pop("telemetry", None)
    material.pop("deterministic_manifest_hash", None)
    runs = material.get("runs")
    if isinstance(runs, list):
        material["runs"] = []
        for run in runs:
            item = copy.deepcopy(run)
            item.pop("wall_seconds", None)
            item.pop("telemetry", None)
            material["runs"].append(item)
    return canonical_hash(material)


def atlas_manifest_hash(payload: dict) -> str:
    material = copy.deepcopy(payload)
    material.pop("telemetry", None)
    material.pop("deterministic_sha256", None)
    enumeration = material.get("candidate_enumeration")
    if isinstance(enumeration, dict):
        enumeration.pop("telemetry", None)
    return canonical_hash(material)


def load_inputs() -> tuple[dict, dict, dict, dict]:
    shell = json.loads(SHELL_PATH.read_text(encoding="utf-8"))
    p1 = json.loads(P1_PATH.read_text(encoding="utf-8"))
    atlas = json.loads(ATLAS_PATH.read_text(encoding="utf-8"))
    shell_hash = shell_manifest_hash(shell)
    p1_hash = p1_manifest_hash(p1)
    atlas_hash = atlas_manifest_hash(atlas)
    assert shell["schema"] == "n25-shell-supercells-v2"
    assert shell["status"] == "exploration"
    assert shell["deterministic_manifest_sha256"] == EXPECTED_SHELL_HASH
    assert shell_hash == EXPECTED_SHELL_HASH
    assert p1["schema"] == "n25-p1-smoke-v1"
    assert p1["status"] == "exploration"
    assert p1["deterministic_manifest_hash"] == EXPECTED_P1_HASH
    assert p1_hash == EXPECTED_P1_HASH
    assert atlas["schema"] == "n25-clipped-voronoi-delaunay-atlas-v3"
    assert atlas["status"] == "exploration"
    assert atlas["deterministic_sha256"] == EXPECTED_ATLAS_HASH
    assert atlas_hash == EXPECTED_ATLAS_HASH
    assert len(p1["runs"]) == 16
    assert atlas["candidate"]["n"] == 25
    atlas_radius = sp.sympify(atlas["candidate"]["target_radius"], locals={"sqrt": sp.sqrt})
    shell_radius = sp.sympify(shell["constants"]["shell_radii_exact"][-1], locals={"sqrt": sp.sqrt})
    assert sp.simplify(atlas_radius - shell_radius) == 0
    return shell, p1, atlas, {"shell": shell_hash, "p1": p1_hash, "atlas": atlas_hash}


def parse_exact(value: str) -> float:
    return float(sp.N(sp.sympify(value, locals={"sqrt": sp.sqrt}), 17))


def atlas_centers(atlas: dict) -> np.ndarray:
    pairs = atlas["candidate"]["centers_exact"]
    return np.asarray([[parse_exact(x), parse_exact(y)] for x, y in pairs], dtype=np.float64)


def role_features(shell: dict) -> tuple[np.ndarray, dict]:
    summary = shell["role_summary"]
    expected = {"O": 1, "A": 8, "B": 8, "C": 8}
    roles = ("O", "A", "B", "C")
    rows = []
    multiplicities = {}
    for role in roles:
        item = summary[role]
        assert item["multiplicity"] == expected[role]
        assert item["expected_multiplicity"] == expected[role]
        assert item["passed_multiplicity"] is True
        rows.append([float(x) for x in item["representative_shell_areas"]] + [float(item["representative_responsibility_angle"])])
        multiplicities[role] = expected[role]
    features = np.asarray(rows, dtype=np.float64)
    return features, multiplicities


def candidate_equalities(features: np.ndarray, multiplicities: dict) -> dict:
    differences = features[1:] - features[0]
    u, singular, vh = np.linalg.svd(differences, full_matrices=True)
    tol = max(differences.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 1.0)
    rank = int(np.sum(singular > tol))
    basis = vh[rank:].T
    residual = float(np.max(np.abs(differences @ basis))) if basis.size else 0.0
    global_feature_sum = features[0] * multiplicities["O"] + features[1] * multiplicities["A"] + features[2] * multiplicities["B"] + features[3] * multiplicities["C"]
    return {
        "difference_matrix": differences.tolist(),
        "rank_tolerance": float(tol),
        "rank": rank,
        "singular_values": singular.tolist(),
        "nullspace_dimension": int(basis.shape[1]),
        "nullspace_basis": basis.tolist(),
        "basis_residual": residual,
        "global_feature_sum": global_feature_sum.tolist(),
        "multiplicities": [multiplicities[r] for r in ("O", "A", "B", "C")],
        "global_feature_identity": "1+8+8+8=25; if role potentials equal U, weighted global feature sum dot w equals 25U",
    }


def make_configs(p1: dict) -> list[dict]:
    configs = []
    for index, run in enumerate(p1["runs"]):
        centers = np.asarray(run["final"]["centers"], dtype=np.float64)
        configs.append({"config_id": f"p1-{index:02d}", "source": "p1-final", "seed": None, "sigma": 0.0, "centers": centers})
    for index, run in enumerate(p1["runs"]):
        sigma = PERTURB_SIGMAS[index % len(PERTURB_SIGMAS)]
        seed = 91000 + index
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, int(round(sigma * 1e6))]))
        centers = np.asarray(run["final"]["centers"], dtype=np.float64) + rng.normal(0.0, sigma, (25, 2))
        configs.append({"config_id": f"perturb-{index:02d}", "source": "p1-final-independent-perturbation", "seed": seed, "sigma": sigma, "centers": centers})
    assert len(configs) == 32
    return configs


def canonical_cycle(values: list[int]) -> list[int]:
    if not values:
        return []
    values = list(values)
    return min(values[i:] + values[:i] for i in range(len(values)))


def _owner(point: np.ndarray, centers: np.ndarray) -> int:
    return int(np.argmin(np.sum((centers - point[None, :]) ** 2, axis=1)))


def boundary_partition(centers: np.ndarray, radius: float) -> dict:
    raw_angles = []
    root_count = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            delta = centers[j] - centers[i]
            A, B = 2.0 * radius * delta
            C = float(np.dot(centers[j], centers[j]) - np.dot(centers[i], centers[i]))
            h = float(math.hypot(A, B))
            if h <= SERVICE_TOL or abs(C) > h + 1e-11:
                continue
            alpha = math.acos(float(np.clip(C / h, -1.0, 1.0)))
            phi = math.atan2(B, A)
            raw_angles.extend((float((phi - alpha) % (2.0 * math.pi)), float((phi + alpha) % (2.0 * math.pi))))
            root_count += 2
    raw_angles.sort()
    events = []
    for angle in raw_angles:
        if not events or abs(angle - events[-1]) > 1e-10:
            events.append(angle)
    if len(events) > 1 and 2.0 * math.pi - events[-1] + events[0] <= 1e-10:
        events.pop()
    intervals = []
    for index, left in enumerate(events):
        right = events[(index + 1) % len(events)]
        if index == len(events) - 1:
            right += 2.0 * math.pi
        if right - left > 1e-12:
            mid = 0.5 * (left + right)
            point = radius * np.array([math.cos(mid), math.sin(mid)])
            intervals.append((left, right, _owner(point, centers)))
    angles = [0.0] * len(centers)
    cycle = []
    for left, right, owner in intervals:
        angles[owner] += right - left
        if not cycle or cycle[-1] != owner:
            cycle.append(owner)
    while len(cycle) > 1 and cycle[0] == cycle[-1]:
        cycle.pop()
    return {"angles": angles, "owner_cycle": canonical_cycle(cycle), "event_count": len(events), "root_count": root_count, "unique_event_count": len(events), "partition_residual": float(sum(right - left for left, right, _ in intervals) - 2.0 * math.pi), "orientation": "CCW"}


def _candidate_membership(point: np.ndarray, site: int, centers: np.ndarray, radius: float) -> bool:
    if float(np.dot(point, point)) > radius * radius + 1e-9:
        return False
    ds = float(np.dot(point - centers[site], point - centers[site]))
    return bool(ds <= float(np.min(np.sum((centers - point[None, :]) ** 2, axis=1))) + 1e-8)


def exhaustive_service(centers: np.ndarray, radius: float, site: int) -> dict:
    ci = centers[site]
    ni = float(np.dot(ci, ci))
    raw, sources = [], []
    for j in range(len(centers)):
        if j == site:
            continue
        cj = centers[j]
        a, b = 2.0 * (cj - ci), float(np.dot(cj, cj) - ni)
        for k in range(j + 1, len(centers)):
            if k == site:
                continue
            ck = centers[k]
            c, d = 2.0 * (ck - ci), float(np.dot(ck, ck) - ni)
            det = float(a[0] * c[1] - a[1] * c[0])
            if abs(det) > SERVICE_TOL:
                raw.append(np.array([(b * c[1] - a[1] * d) / det, (a[0] * d - b * c[0]) / det]))
                sources.append("bisector-bisector")
        aa = float(np.dot(a, a))
        if aa > SERVICE_TOL:
            foot = b * a / aa
            h2 = radius * radius - float(np.dot(foot, foot))
            if h2 >= -1e-10:
                perp = np.array([-a[1], a[0]]) / math.sqrt(aa)
                root = math.sqrt(max(0.0, h2))
                raw.extend((foot + root * perp, foot - root * perp))
                sources.extend(("bisector-target-circle",) * 2)
    norm = math.sqrt(ni)
    if norm > SERVICE_TOL:
        raw.append(-radius * ci / norm)
        sources.append("target-circle-stationary-antipode")
    else:
        raw.extend(radius * np.array(v) for v in ((1, 0), (0, 1), (-1, 0), (0, -1)))
        sources.extend(("target-circle-degenerate-cardinal",) * 4)
    accepted, accepted_sources = [], []
    for point, source in zip(raw, sources):
        if np.all(np.isfinite(point)) and _candidate_membership(point, site, centers, radius) and not any(np.linalg.norm(point - q) <= 1e-8 for q in accepted):
            accepted.append(point)
            accepted_sources.append(source)
    if not accepted:
        return {"max_service_distance": None, "service_candidate_count_raw": len(raw), "service_candidate_count": 0, "service_candidate_source_counts": {}, "most_dangerous_point": None, "most_dangerous_source": None}
    distances = [float(np.linalg.norm(q - ci)) for q in accepted]
    index = int(np.argmax(distances))
    counts = {}
    for source in accepted_sources:
        counts[source] = counts.get(source, 0) + 1
    return {"max_service_distance": distances[index], "service_candidate_count_raw": len(raw), "service_candidate_count": len(accepted), "service_candidate_source_counts": counts, "most_dangerous_point": accepted[index].tolist(), "most_dangerous_source": accepted_sources[index]}


def evaluate_config(centers: np.ndarray, radii: np.ndarray, radius: float, grid: int = GRID) -> tuple[list[dict], dict]:
    angles = (np.arange(grid, dtype=np.float64) + 0.5) * (2.0 * math.pi / grid)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    n = len(centers)
    site_norm = np.sum(centers * centers, axis=1)
    all_intervals: list[tuple[np.ndarray, np.ndarray]] = []
    cells = []
    for site in range(n):
        diff = centers - centers[site]
        a = 2.0 * (directions @ diff.T)
        b = site_norm - site_norm[site]
        upper = np.full(grid, radius, dtype=np.float64)
        lower = np.zeros(grid, dtype=np.float64)
        positive = a > SERVICE_TOL
        negative = a < -SERVICE_TOL
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = b[None, :] / a
        upper = np.minimum(upper, np.min(np.where(positive, ratios, radius), axis=1))
        lower = np.maximum(lower, np.max(np.where(negative, ratios, 0.0), axis=1))
        lower = np.clip(lower, 0.0, radius)
        upper = np.clip(upper, 0.0, radius)
        valid = upper > lower + 1e-13
        all_intervals.append((lower, upper))
        shell_areas = []
        cumulative = []
        for boundary in radii:
            value = 0.5 * np.sum(np.maximum(0.0, np.minimum(upper, boundary) ** 2 - np.minimum(lower, boundary) ** 2)) * (2.0 * math.pi / grid)
            cumulative.append(float(value))
        shell_areas = [cumulative[0]] + [cumulative[i] - cumulative[i - 1] for i in range(1, len(cumulative))]
        lower_points = lower[:, None] * directions
        upper_points = upper[:, None] * directions
        dlow = np.sqrt(np.sum((lower_points - centers[site]) ** 2, axis=1))
        dhigh = np.sqrt(np.sum((upper_points - centers[site]) ** 2, axis=1))
        endpoint_service = np.maximum(dlow, dhigh)
        cells.append({
            "site": site,
            "features": [float(x) for x in shell_areas],
            "cumulative_areas": [float(x) for x in cumulative],
            "shell_areas": [float(x) for x in shell_areas],
            "valid_ray_count": int(np.sum(valid)),
            "midpoint_grid": grid,
            "polar_endpoint_service_estimate": float(np.max(endpoint_service[valid])) if np.any(valid) else None,
        })
    boundary_info = boundary_partition(centers, radius)
    for cell in cells:
        cell["boundary_responsibility_angle"] = boundary_info["angles"][cell["site"]]
        cell["features"].append(cell["boundary_responsibility_angle"])
        cell.update(exhaustive_service(centers, radius, cell["site"]))
        service = cell["max_service_distance"]
        cell["margin"] = None if service is None else float(1.0 - service)
        cell["classification"] = "empty" if service is None else ("numerically-local-feasible" if service <= 1.0 - SERVICE_GUARD else ("borderline" if service <= 1.0 + SERVICE_GUARD else "infeasible"))
        cell["method_status"] = "binary64 exhaustive-for-nondegenerate-candidate enumeration; not a certificate"
    total_area = float(sum(sum(c["shell_areas"]) for c in cells))
    theoretical = math.pi * radius * radius
    shell_totals = [float(sum(c["shell_areas"][k] for c in cells)) for k in range(4)]
    target_shells = [math.pi * radii[0] ** 2] + [math.pi * (radii[k] ** 2 - radii[k - 1] ** 2) for k in range(1, 4)]
    boundary_total = float(sum(c["boundary_responsibility_angle"] for c in cells))
    config_summary = {
        "area_total": total_area,
        "area_expected": theoretical,
        "area_error": float(total_area - theoretical),
        "shell_totals": shell_totals,
        "shell_expected": target_shells,
        "shell_max_abs_error": float(max(abs(a - b) for a, b in zip(shell_totals, target_shells))),
        "shell_midpoint_self_consistency": {"grid": grid, "refine_grid": REFINE_GRID, "independent_of_boundary_and_service": True},
        "boundary_angle_total": boundary_total,
        "boundary_angle_expected": 2.0 * math.pi,
        "boundary_angle_error": float(boundary_total - 2.0 * math.pi),
        "boundary_owner_cycle": boundary_info["owner_cycle"],
        "boundary_orientation": boundary_info["orientation"],
        "boundary_event_count": boundary_info["event_count"],
        "boundary_root_count": boundary_info["root_count"],
        "boundary_unique_event_count": boundary_info["unique_event_count"],
        "boundary_partition_residual": boundary_info["partition_residual"],
        "valid_cells": int(sum(c["classification"] != "empty" for c in cells)),
        "strict_feasible_cells": int(sum(c["classification"] == "numerically-local-feasible" for c in cells)),
        "borderline_cells": int(sum(c["classification"] == "borderline" for c in cells)),
        "infeasible_cells": int(sum(c["classification"] == "infeasible" for c in cells)),
        "empty_cells": int(sum(c["classification"] == "empty" for c in cells)),
        "grid_angular_error_guard": None,
    }
    return cells, config_summary


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "q05": None, "q25": None, "median": None, "q75": None, "q95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"count": int(arr.size), "min": float(arr.min()), "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]), "q95": float(q[4]), "max": float(arr.max())}


def pair_diagnostics(configs: list[dict], U: float, weights: np.ndarray) -> list[dict]:
    diagnostics = []
    for config in configs:
        cells = config["cells"]
        cycle = config["summary"]["boundary_owner_cycle"]
        pairs = []
        for index, left in enumerate(cycle):
            right = cycle[(index + 1) % len(cycle)]
            directed = (int(left), int(right))
            if left != right and directed not in pairs:
                pairs.append(directed)
        records = []
        for left, right in pairs:
            joint = float(2.0 * U - np.dot(weights, np.asarray(cells[left]["features"]) + np.asarray(cells[right]["features"])))
            records.append({"sites": [left, right], "orientation": "CCW", "joint_deficit_2U": joint})
        diagnostics.append({"config_id": config["config_id"], "orientation": "CCW", "pair_count": len(records), "pairs": records})
    return diagnostics


def fit_weights(equalities: dict, configs: list[dict]) -> dict:
    basis = np.asarray(equalities["nullspace_basis"], dtype=np.float64)
    if basis.shape != (5, 2):
        return {"status": "safe-failure-nullspace-dimension", "dimension": int(basis.shape[1] if basis.ndim == 2 else -1)}
    train_refs = [(config["config_id"], cell["site"], cell) for config in configs if config["split"] == "train" for cell in config["cells"] if cell["classification"] == "numerically-local-feasible"]
    holdout_refs = [(config["config_id"], cell["site"], cell) for config in configs if config["split"] == "holdout" for cell in config["cells"] if cell["classification"] == "numerically-local-feasible"]
    train = [item[2] for item in train_refs]
    holdout = [item[2] for item in holdout_refs]
    if len(train) < MIN_STRICT_SAMPLES or len(holdout) < MIN_STRICT_SAMPLES:
        return {"status": "insufficient-samples", "train_count": len(train), "holdout_count": len(holdout)}
    train_x = np.asarray([cell["features"] for cell in train], dtype=np.float64)
    holdout_x = np.asarray([cell["features"] for cell in holdout], dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * math.pi, SCAN_POINTS, endpoint=False)
    best = None
    for theta in angles:
        w = basis @ np.asarray([math.cos(theta), math.sin(theta)])
        U = float(np.dot(w, np.asarray(equalities["candidate_features"], dtype=np.float64)[0]))
        deficits = U - train_x @ w
        score = float(np.min(deficits))
        candidate = (score, theta, w, U)
        if best is None or score > best[0]:
            best = candidate
    assert best is not None
    theta = best[1]
    half_width = 2.0 * math.pi / SCAN_POINTS
    for _ in range(4):
        local = np.linspace(theta - half_width, theta + half_width, REFINE_POINTS)
        candidates = []
        for angle in local:
            w = basis @ np.asarray([math.cos(angle), math.sin(angle)])
            U = float(np.dot(w, np.asarray(equalities["candidate_features"], dtype=np.float64)[0]))
            score = float(np.min(U - train_x @ w))
            candidates.append((score, angle, w, U))
        best = max(candidates, key=lambda item: item[0])
        theta = best[1]
        half_width /= 8.0
    _, theta, weights, U = best
    candidate_potentials = np.asarray([float(np.dot(weights, row)) for row in equalities["candidate_features"]])
    candidate_residual = float(np.max(np.abs(candidate_potentials - U)))
    train_deficits = (U - train_x @ weights).tolist()
    holdout_deficits = (U - holdout_x @ weights).tolist()
    global_feature_sum = np.asarray(equalities["global_feature_sum"], dtype=np.float64)
    global_residual = float(abs(np.dot(weights, global_feature_sum) - 25.0 * U))
    result = {
        "status": "fit-complete",
        "parameterization": "w = nullspace_basis @ [cos(theta), sin(theta)]",
        "l2_norm": float(np.linalg.norm(weights)),
        "scan_resolution": SCAN_POINTS,
        "local_refine_resolution": REFINE_POINTS,
        "local_refine_rounds": 4,
        "theta": float(theta % (2.0 * math.pi)),
        "weights": weights.tolist(),
        "U": U,
        "candidate_potentials": candidate_potentials.tolist(),
        "candidate_equality_residual": candidate_residual,
        "global_potential_sum": float(np.dot(weights, global_feature_sum)),
        "global_target_25U": float(25.0 * U),
        "global_sum_residual": global_residual,
        "train": quantiles(train_deficits),
        "holdout": quantiles(holdout_deficits),
        "most_dangerous_train": {"config_id": train_refs[int(np.argmin(train_deficits))][0], "site": int(train_refs[int(np.argmin(train_deficits))][1]), "deficit": float(np.min(train_deficits))},
        "most_dangerous_holdout": {"config_id": holdout_refs[int(np.argmin(holdout_deficits))][0], "site": int(holdout_refs[int(np.argmin(holdout_deficits))][1]), "deficit": float(np.min(holdout_deficits))},
        "deficit_threshold": DEFICIT_THRESHOLD,
        "train_count": len(train),
        "holdout_count": len(holdout),
    }
    result["train_min_strict"] = result["train"]["min"]
    result["holdout_min_strict"] = result["holdout"]["min"]
    return result


def fit_maximin(equalities: dict, configs: list[dict], selector, label: str) -> dict:
    basis = np.asarray(equalities["nullspace_basis"], dtype=np.float64)
    refs = [(config["config_id"], int(cell["site"]), cell) for config in configs for cell in config["cells"] if selector(config, cell)]
    if basis.shape != (5, 2) or len(refs) < 1:
        return {"status": "insufficient-samples", "label": label, "sample_count": len(refs)}
    x = np.asarray([item[2]["features"] for item in refs], dtype=np.float64)
    candidate = np.asarray(equalities["candidate_features"], dtype=np.float64)
    def score(theta):
        w = basis @ np.asarray([math.cos(theta), math.sin(theta)])
        U = float(np.dot(w, candidate[0]))
        return float(np.min(U - x @ w)), w, U
    best = None
    for theta in np.linspace(0.0, 2.0 * math.pi, SCAN_POINTS, endpoint=False):
        item = (score(float(theta))[0], float(theta))
        if best is None or item[0] > best[0]:
            best = item
    assert best is not None
    theta, half_width = best[1], 2.0 * math.pi / SCAN_POINTS
    for _ in range(4):
        local = np.linspace(theta - half_width, theta + half_width, REFINE_POINTS)
        theta = max((float(a) for a in local), key=lambda a: score(a)[0])
        half_width /= 8.0
    min_score, weights, U = score(theta)
    deficits = U - x @ weights
    potentials = np.asarray([float(np.dot(weights, row)) for row in candidate])
    global_sum = np.asarray(equalities["global_feature_sum"], dtype=np.float64)
    most = int(np.argmin(deficits))
    return {"status": "fit-complete", "label": label, "parameterization": "w = nullspace_basis @ [cos(theta), sin(theta)]", "scan_resolution": SCAN_POINTS, "local_refine_resolution": REFINE_POINTS, "local_refine_rounds": 4, "theta": float(theta % (2.0 * math.pi)), "weights": weights.tolist(), "U": U, "maximin_min": min_score, "candidate_potentials": potentials.tolist(), "candidate_equality_residual": float(np.max(np.abs(potentials - U))), "global_potential_sum": float(np.dot(weights, global_sum)), "global_target_25U": float(25.0 * U), "global_sum_residual": float(abs(np.dot(weights, global_sum) - 25.0 * U)), "assessment": quantiles(deficits.tolist()), "min_deficit": float(np.min(deficits)), "max_deficit": float(np.max(deficits)), "sample_count": len(refs), "most_dangerous": {"config_id": refs[most][0], "site": refs[most][1], "deficit": float(deficits[most])}, "deficit_threshold": DEFICIT_THRESHOLD}


def fit_status(fit: dict) -> str:
    if fit.get("status") != "fit-complete":
        return "sampled-inconclusive"
    value = float(fit["min_deficit"])
    if value > DEFICIT_THRESHOLD:
        return "sampled-positive-separator"
    if value < -DEFICIT_THRESHOLD:
        return "sampled-no-separator"
    return "sampled-inconclusive"


def report_legacy(payload: dict) -> str:
    fit = payload["fit"]
    eq = payload["candidate_equality"]
    lines = [
        "# n=25 P2 五特征势函数发现实验", "",
        "> 状态：**exploration**。本实验不是 fixed-radius certificate、Stage A、Stage B 或 global optimality proof。", "",
        "## 复现", "", "```text", payload["reproduction"]["command"], "```", "",
        f"产物 deterministic manifest SHA-256：`{payload['deterministic_manifest_hash']}`。",
        f"输入 shell hash 校验：`{payload['input_hashes']['shell']['recomputed']}` == expected；P1：`{payload['input_hashes']['p1']['recomputed']}` == expected。", "",
        "## 结果", "",
        f"five-feature-family-status：**{payload['five-feature-family-status']}**；success_signal：`{payload['success_signal']}`。",
        f"配置数：{len(payload['configs'])}；cell 样本总数：{payload['sample_summary']['total_cells']}；严格可行：{payload['sample_summary']['strict_feasible_cells']}；borderline：{payload['sample_summary']['borderline_cells']}；infeasible：{payload['sample_summary']['infeasible_cells']}；empty：{payload['sample_summary']['empty_cells']}。", "",
        "角色标签只用于候选 O/A/B/C baseline；竞争 cells 不预贴 O/A/B/C 标签。局部筛选、面积、边界角度和 service distance 均来自 binary64 polar midpoint grid，不是证书。", "",
        "## 候选等式与 nullspace", "",
        f"差分矩阵 rank={eq['rank']}；singular values={eq['singular_values']}；nullspace dimension={eq['nullspace_dimension']}；basis residual={eq['basis_residual']}。",
        f"候选 multiplicities={eq['multiplicities']}，满足 1+8+8+8=25；候选相等势时全局加权 feature sum 必为 25U。", "",
        "## 最优数值权重", "",
        f"weights={fit.get('weights')}；L2 norm={fit.get('l2_norm')}；U={fit.get('U')}；theta={fit.get('theta')}。",
        f"train strict deficit min={fit.get('train_min_strict')}；holdout strict deficit min={fit.get('holdout_min_strict')}。",
        f"most dangerous train={fit.get('most_dangerous_train')}；holdout={fit.get('most_dangerous_holdout')}。",
        f"train quantiles={fit.get('train')}；holdout quantiles={fit.get('holdout')}。",
        f"candidate equality residual={fit.get('candidate_equality_residual')}；global sum residual={fit.get('global_sum_residual')}。", "",
        "## 语义边界与限制", "",
        "本实验不证明不可兼任，不证明 BC 超单元局部不等式，不证明固定半径刚性，也不进入 Stage A、Stage B 或全局最优性。Voronoi cell 用 midpoint angular grid 的半平面径向区间积分；每个 cell 的 max service distance 只由采样射线端点估计，空 cell 单列，`numerically-local-feasible` 不是 certified admissible。", "",
        f"失败/解释：{payload['interpretation']}", "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    shell, p1, atlas, hashes = load_inputs()
    centers0 = atlas_centers(atlas)
    radius = float(sp.N(sp.sympify(atlas["candidate"]["target_radius"], locals={"sqrt": sp.sqrt}), 17))
    radii = np.asarray([parse_exact(x) for x in shell["constants"]["shell_radii_exact"]], dtype=np.float64)
    assert centers0.shape == (25, 2)
    assert radii.shape == (4,) and np.all(np.diff(radii) > 0)
    features, multiplicities = role_features(shell)
    equalities = candidate_equalities(features, multiplicities)
    equalities["candidate_features"] = features.tolist()
    configs = make_configs(p1)
    for index, config in enumerate(configs):
        config["split"] = "train" if index % 2 == 0 else "holdout"
        cells, summary = evaluate_config(config["centers"], radii, radius)
        config["centers"] = config["centers"].tolist()
        config["centers_hash"] = canonical_hash(config["centers"])
        config["cells"] = cells
        config["summary"] = summary
    all_cells = [cell for config in configs for cell in config["cells"]]
    assert all(abs(config["summary"]["boundary_angle_error"]) <= 1e-12 for config in configs)
    assert all(config["summary"]["shell_max_abs_error"] <= 1e-12 for config in configs)
    assert all(abs(config["summary"]["area_error"]) <= 1e-12 for config in configs)
    sample_summary = {
        "total_cells": len(all_cells),
        "strict_feasible_cells": sum(c["classification"] == "numerically-local-feasible" for c in all_cells),
        "borderline_cells": sum(c["classification"] == "borderline" for c in all_cells),
        "infeasible_cells": sum(c["classification"] == "infeasible" for c in all_cells),
        "empty_cells": sum(c["classification"] == "empty" for c in all_cells),
        "config_count": len(configs),
        "train_config_count": sum(c["split"] == "train" for c in configs),
        "holdout_config_count": sum(c["split"] == "holdout" for c in configs),
        "classification_definition": "strict iff max_service_distance <= 1-SERVICE_GUARD; borderline iff within +/- SERVICE_GUARD; empty has no valid sampled ray interval",
        "service_guard": SERVICE_GUARD,
    }
    fit = fit_weights(equalities, configs)
    if fit.get("status") == "fit-complete":
        diagnostics = pair_diagnostics(configs, fit["U"], np.asarray(fit["weights"], dtype=np.float64))
        fit_ok = (fit["candidate_equality_residual"] <= 1e-10 and fit["global_sum_residual"] <= 1e-10 and fit["train_min_strict"] > DEFICIT_THRESHOLD and fit["holdout_min_strict"] > DEFICIT_THRESHOLD)
        enough = fit["train_count"] >= MIN_STRICT_SAMPLES and fit["holdout_count"] >= MIN_STRICT_SAMPLES
        status = "sampled-success-signal" if fit_ok and enough else "sampled-failure"
    else:
        diagnostics = []
        status = "insufficient-samples" if fit.get("status") == "insufficient-samples" else "numerical-signal"
    interpretation = ("No global or local proof is obtained. The reported status is only a config-split sampled signal; "
                      "strict deficits use the chosen sampled cells and binary64 numerical geometry.")
    payload = {
        "schema": "n25-p2-potential-search-v1",
        "status": "exploration",
        "five-feature-family-status": status,
        "success_signal": status == "sampled-success-signal",
        "input_hashes": {
            "shell": {"path": "data/n25_shell_supercells.json", "expected": EXPECTED_SHELL_HASH, "recomputed": hashes["shell"], "passed": hashes["shell"] == EXPECTED_SHELL_HASH},
            "p1": {"path": "data/n25_p1_smoke.json", "expected": EXPECTED_P1_HASH, "recomputed": hashes["p1"], "passed": hashes["p1"] == EXPECTED_P1_HASH},
            "p0_atlas": {"path": "data/n25_voronoi_atlas.json", "schema": atlas["schema"], "status": atlas["status"], "center_source": "candidate.centers_exact", "center_count": int(len(centers0))},
        },
        "reference": {"R": radius, "t0_t3": radii.tolist(), "center_count": 25, "centers": centers0.tolist(), "centers_hash": canonical_hash(centers0.tolist())},
        "parameters": {"grid": GRID, "boundary_grid": BOUNDARY_GRID, "service_guard": SERVICE_GUARD, "deficit_threshold": DEFICIT_THRESHOLD, "min_strict_samples": MIN_STRICT_SAMPLES, "scan_points": SCAN_POINTS, "refine_points": REFINE_POINTS, "perturb_sigmas": list(PERTURB_SIGMAS), "binary64": True},
        "candidate_equality": equalities,
        "configs": configs,
        "sample_summary": sample_summary,
        "fit": fit,
        "bc_cb_diagnostics": diagnostics,
        "interpretation": interpretation,
        "limitations": [
            "Exploration only; sampled numerical evidence is not a certificate.",
            "Competitive cells are not assigned O/A/B/C labels.",
            "No proof of non-coassignability, BC supercell inequality, fixed-radius rigidity, Stage A, Stage B, or global optimality.",
            "Voronoi radial intervals, areas, boundary angles, and max service distances use binary64 midpoint angular sampling.",
            "A numerically-local-feasible cell is not certified admissible; empty cells are excluded and reported separately.",
            "Config-level split prevents cell leakage, but the finite sampled configuration family is not exhaustive.",
        ],
        "reproduction": {"command": ".venv/bin/python src/n25_p2_potential_search.py --smoke", "python": platform.python_version(), "numpy": np.__version__, "sympy": sp.__version__},
        "validation": {"input_schema_and_hashes_checked": True, "internal_assertions": True, "runtime_seconds": None},
        "deterministic_manifest_hash": "",
    }
    manifest = copy.deepcopy(payload)
    manifest.pop("deterministic_manifest_hash", None)
    manifest.get("validation", {}).pop("runtime_seconds", None)
    payload["deterministic_manifest_hash"] = canonical_hash(manifest)
    payload["validation"]["runtime_seconds"] = float(time.perf_counter() - started)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    DOC_PATH.write_text(report(payload), encoding="utf-8")
    assert payload["status"] == "exploration"
    assert payload["input_hashes"]["shell"]["passed"] and payload["input_hashes"]["p1"]["passed"]
    assert len(payload["configs"]) == 32 and len(all_cells) == 800
    assert payload["candidate_equality"]["multiplicities"] == [1, 8, 8, 8]
    assert payload["candidate_equality"]["nullspace_dimension"] == 2
    assert all(math.isfinite(float(x)) for c in configs for cell in c["cells"] for x in cell["features"])
    print(json.dumps({"output": str(OUT_PATH), "samples": len(all_cells), "classification": status, "nullspace": 2, "weights": fit.get("weights"), "U": fit.get("U"), "train_min": fit.get("train_min_strict"), "holdout_min": fit.get("holdout_min_strict"), "manifest_hash": payload["deterministic_manifest_hash"], "runtime_seconds": payload["validation"]["runtime_seconds"]}, ensure_ascii=False))


def _stable(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def assess_fit(fit: dict, configs: list[dict], selector, label: str) -> dict:
    if fit.get("status") != "fit-complete":
        return {"label": label, "status": "sampled-inconclusive", "sample_count": 0, "min_deficit": None}
    weights = np.asarray(fit["weights"], dtype=np.float64)
    U = float(fit["U"])
    refs = [(config["config_id"], int(cell["site"]), cell) for config in configs for cell in config["cells"] if selector(config, cell)]
    deficits = np.asarray([U - float(np.dot(weights, cell["features"])) for _, _, cell in refs])
    index = int(np.argmin(deficits))
    value = float(deficits[index]) if len(deficits) else None
    status = "sampled-positive-separator" if value is not None and value > DEFICIT_THRESHOLD else ("sampled-no-separator" if value is not None and value < -DEFICIT_THRESHOLD else "sampled-inconclusive")
    return {"label": label, "status": status, "sample_count": len(refs), "min_deficit": value, "max_deficit": float(np.max(deficits)) if len(deficits) else None, "assessment": quantiles(deficits.tolist()), "most_dangerous": None if not refs else {"config_id": refs[index][0], "site": refs[index][1], "deficit": value}}


def refinement_for_fit(fit: dict, configs: list[dict], radii: np.ndarray, radius: float, label: str, danger_override: dict | None = None) -> dict:
    danger = danger_override or fit.get("most_dangerous")
    if not danger:
        return {"label": label, "status": "not-run"}
    config = next(item for item in configs if item["config_id"] == danger["config_id"])
    refined_cells, refined_summary = evaluate_config(np.asarray(config["centers"], dtype=np.float64), radii, radius, grid=REFINE_GRID)
    cell = refined_cells[int(danger["site"])]
    weights = np.asarray(fit["weights"], dtype=np.float64)
    refined_deficit = float(fit["U"] - np.dot(weights, np.asarray(cell["features"], dtype=np.float64)))
    old_bucket = "positive" if danger["deficit"] > DEFICIT_THRESHOLD else ("negative" if danger["deficit"] < -DEFICIT_THRESHOLD else "inconclusive")
    new_bucket = "positive" if refined_deficit > DEFICIT_THRESHOLD else ("negative" if refined_deficit < -DEFICIT_THRESHOLD else "inconclusive")
    return {"label": label, "config_id": danger["config_id"], "site": int(danger["site"]), "grid": GRID, "refine_grid": REFINE_GRID, "original_deficit": float(danger["deficit"]), "refined_deficit": refined_deficit, "deficit_delta": refined_deficit - float(danger["deficit"]), "original_classification": next(c["classification"] for c in config["cells"] if c["site"] == danger["site"]), "refined_classification": cell["classification"], "original_bucket": old_bucket, "refined_bucket": new_bucket, "changed": old_bucket != new_bucket or cell["classification"] != next(c["classification"] for c in config["cells"] if c["site"] == danger["site"]), "refined_features": cell["features"], "refined_summary": {"shell_max_abs_error": refined_summary["shell_max_abs_error"], "boundary_partition_residual": refined_summary["boundary_partition_residual"]}}


def report(payload: dict) -> str:
    lines = [
        "# n=25 P2 五特征势函数发现实验 v2", "",
        "> 状态：**exploration**；不是 fixed-radius certificate、Stage A、Stage B 或 global optimality proof。", "",
        "## 修正与复现", "",
        "旧 v1 manifest `24ec86e155ee98d75feaaa989ff2b848d925b9ee9561707b21883c53f4221dee` 作废。v2 将 service 改为 exhaustive-for-nondegenerate-candidate 枚举，将 boundary 改为完整等距事件分区，并把 train-only 与 combined maximin 分开。", "",
        "```text", payload["reproduction"]["command"], "```",
        f"v2 manifest：`{payload['deterministic_manifest_hash']}`；输入 hashes：{_stable(payload['input_hashes'])}。", "",
        "## 结果", "",
        f"classification counts：{_stable(payload['sample_summary'])}",
        f"train-weight-holdout-status：**{payload['train-weight-holdout-status']}**；five-feature-family-status：**{payload['five-feature-family-status']}**。",
        "five-feature-family-status 仅由 combined fit 的 min_deficit 与 deficit_threshold 决定；refinement 只记录离散化敏感性，不覆盖该 family 状态。",
        f"train min={payload['fits']['train'].get('min_deficit')}；holdout assessment min={payload['fits']['train_weight_holdout'].get('min_deficit')}；combined min={payload['fits']['combined'].get('min_deficit')}。", "",
        "## Fits 与 refinement", "",
        f"train fit：{_stable(payload['fits']['train'])}", f"train-weight holdout：{_stable(payload['fits']['train_weight_holdout'])}", f"combined fit：{_stable(payload['fits']['combined'])}", f"refinement：{_stable(payload['refinement'])}", "",
        "## 几何语义边界", "",
        "壳层面积独立使用 midpoint polar integration（GRID=8192，并记录 4x refinement）；boundary angle/cycle 使用 300 site-pair 的全部 trig 等距事件，按 CCW 区间 midpoint owner 分区；service 使用 bisector-bisector、bisector-target-circle 和 target-circle stationary antipode 候选，均为 binary64 exhaustive-for-nondegenerate-candidate enumeration，不是证书。", "",
        f"reference calibration：{_stable(payload['reference_calibration'])}", "",
        f"限制：{_stable(payload['limitations'])}", "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="n=25 P2 finite sampled potential search")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    shell, p1, atlas, hashes = load_inputs()
    centers0 = atlas_centers(atlas)
    radius = float(sp.N(sp.sympify(atlas["candidate"]["target_radius"], locals={"sqrt": sp.sqrt}), 17))
    radii = np.asarray([parse_exact(x) for x in shell["constants"]["shell_radii_exact"]], dtype=np.float64)
    assert centers0.shape == (25, 2) and radii.shape == (4,) and np.all(np.diff(radii) > 0)
    features, multiplicities = role_features(shell)
    equalities = candidate_equalities(features, multiplicities)
    equalities["candidate_features"] = features.tolist()
    configs = make_configs(p1)
    for index, config in enumerate(configs):
        config["split"] = "train" if index % 2 == 0 else "holdout"
        cells, summary = evaluate_config(config["centers"], radii, radius)
        config["centers"] = config["centers"].tolist()
        config["centers_hash"] = canonical_hash(config["centers"])
        config["cells"], config["summary"] = cells, summary
    all_cells = [cell for config in configs for cell in config["cells"]]
    assert all(abs(c["summary"]["boundary_angle_error"]) <= 1e-12 and abs(c["summary"]["boundary_partition_residual"]) <= 1e-12 for c in configs)
    assert all(c["summary"]["shell_max_abs_error"] <= 1e-12 and abs(c["summary"]["area_error"]) <= 1e-12 for c in configs)
    reference_cells, reference_summary = evaluate_config(centers0, radii, radius)
    assert abs(reference_cells[9]["boundary_responsibility_angle"] - features[2, 4]) < 1e-5
    assert abs(reference_cells[17]["boundary_responsibility_angle"] - features[3, 4]) < 1e-5
    reference_service_error = max(abs(float(c["max_service_distance"]) - 1.0) for c in reference_cells)
    assert reference_service_error < 1e-8
    strict = lambda config, cell: cell["classification"] == "numerically-local-feasible"
    train = lambda config, cell: config["split"] == "train" and strict(config, cell)
    holdout = lambda config, cell: config["split"] == "holdout" and strict(config, cell)
    fit_train = fit_maximin(equalities, configs, train, "train-only")
    holdout_assessment = assess_fit(fit_train, configs, holdout, "train-weight-holdout")
    fit_combined = fit_maximin(equalities, configs, strict, "combined-all-local-feasible")
    train_assessment = assess_fit(fit_train, configs, train, "train-weight-train")
    combined_assessment = assess_fit(fit_combined, configs, strict, "combined-assessment")
    fits = {"train": fit_train, "train_weight_train": train_assessment, "train_weight_holdout": holdout_assessment, "combined": fit_combined, "combined_assessment": combined_assessment}
    refinement = {"train_weight_holdout": refinement_for_fit(fit_train, configs, radii, radius, "train-weight holdout danger", holdout_assessment.get("most_dangerous")), "combined": refinement_for_fit(fit_combined, configs, radii, radius, "combined danger")}
    combined_status = fit_status(fit_combined)
    diagnostics = pair_diagnostics(configs, float(fit_combined.get("U", 0.0)), np.asarray(fit_combined.get("weights", [0.0] * 5), dtype=np.float64)) if fit_combined.get("status") == "fit-complete" else []
    sample_summary = {"total_cells": len(all_cells), "strict_feasible_cells": sum(c["classification"] == "numerically-local-feasible" for c in all_cells), "borderline_cells": sum(c["classification"] == "borderline" for c in all_cells), "infeasible_cells": sum(c["classification"] == "infeasible" for c in all_cells), "empty_cells": sum(c["classification"] == "empty" for c in all_cells), "config_count": len(configs), "train_config_count": sum(c["split"] == "train" for c in configs), "holdout_config_count": sum(c["split"] == "holdout" for c in configs), "classification_definition": "strict iff exhaustive service <= 1-SERVICE_GUARD; borderline iff within +/- SERVICE_GUARD; no angular-spacing guarantee", "service_guard": SERVICE_GUARD}
    interpretation = "Finite sampled signal only; no global/local proof, no Stage A/B, no fixed-radius rigidity, and no optimality proof."
    payload = {"schema": "n25-p2-potential-search-v2", "status": "exploration", "supersedes_v1_manifest": "24ec86e155ee98d75feaaa989ff2b848d925b9ee9561707b21883c53f4221dee", "five-feature-family-status": combined_status, "success_signal": combined_status == "sampled-positive-separator", "train-weight-holdout-status": holdout_assessment["status"], "input_hashes": {"shell": {"path": "data/n25_shell_supercells.json", "expected": EXPECTED_SHELL_HASH, "recomputed": hashes["shell"], "passed": hashes["shell"] == EXPECTED_SHELL_HASH}, "p1": {"path": "data/n25_p1_smoke.json", "expected": EXPECTED_P1_HASH, "recomputed": hashes["p1"], "passed": hashes["p1"] == EXPECTED_P1_HASH}, "p0_atlas": {"path": "data/n25_voronoi_atlas.json", "expected": EXPECTED_ATLAS_HASH, "recomputed": hashes["atlas"], "passed": hashes["atlas"] == EXPECTED_ATLAS_HASH, "schema": atlas["schema"], "status": atlas["status"], "center_count": 25, "R_equals_shell_t3": True}}, "reference": {"R": radius, "t0_t3": radii.tolist(), "center_count": 25, "centers": centers0.tolist(), "centers_hash": canonical_hash(centers0.tolist())}, "reference_calibration": {"service_method": "exhaustive-for-nondegenerate-candidate", "max_abs_service_minus_one": reference_service_error, "boundary_event_count": reference_summary["boundary_event_count"], "boundary_root_count": reference_summary["boundary_root_count"], "B_angle_error": float(reference_cells[9]["boundary_responsibility_angle"] - features[2, 4]), "C_angle_error": float(reference_cells[17]["boundary_responsibility_angle"] - features[3, 4])}, "parameters": {"grid": GRID, "refine_grid": REFINE_GRID, "boundary_grid": None, "boundary_event_enumeration": "all 300 site-pairs", "service_guard": SERVICE_GUARD, "service_tolerance": SERVICE_TOL, "deficit_threshold": DEFICIT_THRESHOLD, "min_strict_samples": MIN_STRICT_SAMPLES, "scan_points": SCAN_POINTS, "refine_points": REFINE_POINTS, "perturb_sigmas": list(PERTURB_SIGMAS), "binary64": True}, "candidate_equality": equalities, "configs": configs, "sample_summary": sample_summary, "fits": fits, "refinement": refinement, "bc_cb_diagnostics": diagnostics, "interpretation": interpretation, "limitations": ["Exploration only; all statuses are finite sampled signals, not certificates.", "Shell areas use independent midpoint polar integration; boundary and service use separate exhaustive numerical enumerations.", "Service enumeration is exhaustive only for nondegenerate floating-point candidate geometry and does not certify continuum extrema.", "No proof of non-coassignability, BC inequality, Stage A, Stage B, fixed-radius rigidity, or global optimality.", "The finite sampled configuration family is not exhaustive; parent and derived configurations retain the same split."], "reproduction": {"command": ".venv/bin/python src/n25_p2_potential_search.py --smoke", "python": platform.python_version(), "numpy": np.__version__, "sympy": sp.__version__}, "validation": {"input_schema_and_hashes_checked": True, "internal_assertions": True, "runtime_seconds": None}, "deterministic_manifest_hash": ""}
    manifest = copy.deepcopy(payload)
    manifest.pop("deterministic_manifest_hash", None)
    manifest["validation"].pop("runtime_seconds", None)
    payload["deterministic_manifest_hash"] = canonical_hash(manifest)
    payload["validation"]["runtime_seconds"] = float(time.perf_counter() - started)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    DOC_PATH.write_text(report(payload), encoding="utf-8")
    assert payload["five-feature-family-status"] == fit_status(fit_combined)
    assert payload["input_hashes"]["p0_atlas"]["passed"] and all(d["orientation"] == "CCW" for d in diagnostics)
    assert len(configs) == 32 and len(all_cells) == 800 and payload["candidate_equality"]["nullspace_dimension"] == 2
    print(_stable({"classification_counts": {k: sample_summary[k] for k in ("strict_feasible_cells", "borderline_cells", "infeasible_cells", "empty_cells")}, "reference_service_error": reference_service_error, "train_min": train_assessment["min_deficit"], "holdout_min": holdout_assessment["min_deficit"], "combined_min": fit_combined.get("min_deficit"), "train_weight_holdout_status": payload["train-weight-holdout-status"], "status": payload["five-feature-family-status"], "manifest_hash": payload["deterministic_manifest_hash"], "runtime_seconds": payload["validation"]["runtime_seconds"]}))


if __name__ == "__main__":
    main()
