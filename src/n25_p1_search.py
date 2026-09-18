"""P1 n=25 fixed-radius unrestricted numerical exploration.

This is a bounded floating-point search. It is not a certificate and does not
enter finite-point Stage B.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from config25 import R25, centers25
from geom import covering_radius
from hole import hole_candidates
from search import alternate, polar_net

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "n25_p1_smoke.json"
DOC_PATH = ROOT / "docs" / "n25_p1_search.md"
SIGMAS = (1e-6, 1e-4, 1e-2, 5e-2)
SEEDS = (11, 99)
RESTARTS = 2
ROUNDS = 12
N_R = 16
MAX_HOLE_EVALS = 250
POLL_TRIES = 2
ACTIVE_TOL = 1e-7
STRICT_TOL = 1e-8


def digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                        allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_cycle(values):
    if not values:
        return []
    values = list(values)
    rotations = [values[i:] + values[:i] for i in range(len(values))]
    reversed_values = list(reversed(values))
    rotations.extend(reversed_values[i:] + reversed_values[:i] for i in range(len(values)))
    return min(rotations)


def boundary_candidate_fingerprint(records: list[dict]) -> dict:
    boundary_records = [r for r in records if abs(r["radial_residual"]) <= 5e-7]
    boundary_records.sort(key=lambda r: math.atan2(r["point"][1], r["point"][0]) % (2 * math.pi))
    pair_owners = [tuple(r["nearest_sites"]) for r in boundary_records
                   if r["kind"] == "boundary-bisector"]
    switches = []
    if pair_owners:
        for i, owner in enumerate(pair_owners):
            if owner != pair_owners[i - 1]:
                switches.append([int(i), list(owner)])
    return {"boundary_candidate_switches": switches, "status": "floating-point fingerprint"}


def sampled_boundary_owner_cycle(C: np.ndarray) -> list[int]:
    angles = np.linspace(0.0, 2.0 * math.pi, 256, endpoint=False)
    boundary_points = np.stack((R25 * np.cos(angles), R25 * np.sin(angles)), axis=1)
    owner_samples = np.argmin(np.linalg.norm(boundary_points[:, None, :] - C[None, :, :], axis=2), axis=1)
    owner_cycle = [int(owner_samples[0])]
    for owner in owner_samples[1:]:
        owner = int(owner)
        if owner != owner_cycle[-1]:
            owner_cycle.append(owner)
    while len(owner_cycle) > 1 and owner_cycle[0] == owner_cycle[-1]:
        owner_cycle.pop()
    return canonical_cycle(owner_cycle)


def topology_signature(C: np.ndarray, records: list[dict]) -> dict:
    faces = sorted({tuple(r["nearest_sites"]) for r in records
                    if r["kind"] == "triple-circumcenter"
                    and len(r["nearest_sites"]) == 3})
    edges = set()
    for face in faces:
        for i in range(3):
            for j in range(i + 1, len(face)):
                edges.add(tuple(sorted((face[i], face[j]))))
    for r in records:
        if r["kind"] == "boundary-bisector" and len(r["nearest_sites"]) >= 2:
            owners = tuple(r["nearest_sites"])
            for i in range(len(owners)):
                for j in range(i + 1, len(owners)):
                    edges.add(tuple(sorted((owners[i], owners[j]))))
    edge_list = sorted(edges)
    sites = len(C)
    edges_count = len(edge_list)
    faces_count = len(faces)
    return {
        "delaunay_edges": [list(x) for x in edge_list],
        "delaunay_faces": [list(x) for x in faces],
        "boundary_owner_cycle": sampled_boundary_owner_cycle(C),
        "counts": {"sites": sites, "E": edges_count, "F": faces_count,
                   "euler": sites - edges_count + faces_count},
    }


class HoleBudget:
    def __init__(self, maximum: int):
        self.maximum = int(maximum)
        self.count = 0
        self.stop_reason = None

    def evaluate(self, C: np.ndarray, R: float) -> tuple[float, np.ndarray, list[dict]]:
        if self.count >= self.maximum:
            self.stop_reason = "max_hole_evals"
            raise RuntimeError("hole evaluation budget exhausted")
        self.count += 1
        records = hole_candidates(C, R)
        if not records:
            return 0.0, np.zeros(2), records
        index = int(np.argmax([r["value"] for r in records]))
        record = records[index]
        return float(record["value"]), np.asarray(record["point"], dtype=float), records


def active_records(records: list[dict], H: float) -> list[dict]:
    return [r for r in records if r["value"] >= H - ACTIVE_TOL]


def trajectory_record(round_no: int, H: float, q: np.ndarray, records: list[dict], C: np.ndarray,
                      eval_count: int, event: str) -> dict:
    active = active_records(records, H)
    topology = topology_signature(C, records)
    topology["hash"] = digest(topology)
    active_payload = [{
        "kind": r["kind"],
        "point": r["point"],
        "defining_sites": r["defining_sites"],
        "nearest_sites": r["nearest_sites"],
        "value": r["value"],
        "radial_residual": r["radial_residual"],
        "status": r.get("status", "floating-point exploratory"),
    } for r in active]
    active_kind_owners = sorted({(r["kind"], tuple(r["nearest_sites"])) for r in active})
    active_kind_owners = [[kind, list(owners)] for kind, owners in active_kind_owners]
    return {
        "round": int(round_no),
        "event": event,
        "H": float(H),
        "deepest_point": [float(q[0]), float(q[1])],
        "hole_evals": int(eval_count),
        "active_holes": active_payload,
        "active_kind_owners": active_kind_owners,
        "active_hash": digest(active_payload),
        "active_kind_owners_hash": digest(active_kind_owners),
        "raw_boundary_fingerprint": boundary_candidate_fingerprint(records),
        "topology": topology,
    }


def run_one(base_seed: int, sigma_index: int, sigma: float, restart: int,
            reference: np.ndarray) -> dict:
    seed_sequence = np.random.SeedSequence([int(base_seed), int(sigma_index), int(restart)])
    rng = np.random.default_rng(seed_sequence)
    C = reference + rng.normal(scale=sigma, size=reference.shape)
    budget = HoleBudget(MAX_HOLE_EVALS)
    P = polar_net(R25, N_R)
    trajectory = []
    started = time.perf_counter()
    stop_reason = "rounds_complete"
    try:
        H, q, records = budget.evaluate(C, R25)
        P = np.vstack((P, q))
        trajectory.append(trajectory_record(0, H, q, records, C, budget.count, "initial"))
        best_H, best_C = H, C.copy()
        best_q, best_records = q.copy(), copy.deepcopy(records)
        for round_no in range(1, ROUNDS + 1):
            C = alternate(P, C, iters=4)
            H, q, records = budget.evaluate(C, R25)
            if H < best_H:
                best_H, best_C = H, C.copy()
                best_q, best_records = q.copy(), copy.deepcopy(records)
            P = np.vstack((P, q))
            for record in active_records(records, H)[:24]:
                P = np.vstack((P, np.asarray(record["point"], dtype=float)))

            for poll_no in range(POLL_TRIES):
                if budget.count >= MAX_HOLE_EVALS:
                    stop_reason = "max_hole_evals"
                    break
                trial = C.copy()
                site = int(rng.integers(len(C)))
                step = (0.006 * (0.65 ** poll_no)) / (1.0 + 0.15 * round_no)
                trial[site] += rng.normal(scale=step, size=2)
                trial_H, trial_q, trial_records = budget.evaluate(trial, R25)
                if trial_H < H:
                    C, H, q, records = trial, trial_H, trial_q, trial_records
                    if H < best_H:
                        best_H, best_C = H, C.copy()
                        best_q, best_records = q.copy(), copy.deepcopy(records)
                    P = np.vstack((P, q))
            if budget.count >= MAX_HOLE_EVALS:
                stop_reason = "max_hole_evals"
            trajectory.append(trajectory_record(round_no, H, q, records, C, budget.count,
                                                "cutting-plane-poll"))
            if stop_reason == "max_hole_evals":
                break
    except RuntimeError:
        stop_reason = "max_hole_evals"
    if budget.count >= MAX_HOLE_EVALS:
        stop_reason = "max_hole_evals"

    for index, step in enumerate(trajectory):
        if index == 0:
            step["signature_change"] = None
        else:
            previous = trajectory[index - 1]
            changed = []
            if step["active_hash"] != previous["active_hash"]:
                changed.append("active_hash")
            if step["active_kind_owners_hash"] != previous["active_kind_owners_hash"]:
                changed.append("active_kind_owners_hash")
            if step["topology"]["hash"] != previous["topology"]["hash"]:
                changed.append("topology_hash")
            step["signature_change"] = {"from_round": previous["round"],
                                         "from_active_hash": previous["active_hash"],
                                         "from_active_kind_owners_hash": previous["active_kind_owners_hash"],
                                         "from_topology_hash": previous["topology"]["hash"],
                                         "changed": changed} if changed else None

    if budget.count < MAX_HOLE_EVALS:
        final_H, final_q, final_records = budget.evaluate(best_C, R25)
    else:
        final_H, final_q, final_records = best_H, best_q, best_records
    final_topology = topology_signature(best_C, final_records)
    final_topology["hash"] = digest(final_topology)
    final_active = active_records(final_records, final_H)
    final_active_kind_owners = [[kind, list(owners)] for kind, owners in
                                sorted({(r["kind"], tuple(r["nearest_sites"])) for r in final_active})]
    radius = float(covering_radius(best_C))
    elapsed = time.perf_counter() - started
    return {
        "input": {"base_seed": base_seed, "sigma_index": sigma_index, "sigma": sigma,
                  "restart": restart, "seed_sequence": seed_sequence.entropy,
                  "seed_spawn_key": list(seed_sequence.spawn_key)},
        "budget": {"max_hole_evals": MAX_HOLE_EVALS, "hole_evals": budget.count,
                   "stop_reason": stop_reason, "poll_tries_per_round": POLL_TRIES},
        "trajectory": trajectory,
        "final": {"H": float(final_H), "deepest_point": [float(x) for x in final_q],
                  "centers": best_C.tolist(), "covering_radius": radius,
                  "candidate_count": len(final_records),
                  "candidate_hash": digest(final_records),
                  "hole_candidates": final_records,
                  "active_holes": final_active,
                  "active_hash": digest(final_active),
                  "active_kind_owners": final_active_kind_owners,
                  "active_kind_owners_hash": digest(final_active_kind_owners),
                  "raw_boundary_fingerprint": boundary_candidate_fingerprint(final_records),
                  "topology": final_topology},
        "wall_seconds": elapsed,
    }


def congruence(candidate: np.ndarray, reference: np.ndarray) -> dict:
    best = None
    for reflected in (False, True):
        X = candidate.copy()
        if reflected:
            X[:, 1] *= -1.0
        Q = np.eye(2)
        for _ in range(12):
            transformed = X @ Q
            rows, cols = linear_sum_assignment(np.linalg.norm(transformed[:, None, :] - reference[None, :, :], axis=2) ** 2)
            assignment = np.empty(len(X), dtype=int)
            assignment[rows] = cols
            Y = reference[assignment]
            M = transformed.T @ Y
            U, _, Vt = np.linalg.svd(M)
            delta = U @ Vt
            Q = Q @ delta
        transformed = X @ Q
        rows, cols = linear_sum_assignment(np.linalg.norm(transformed[:, None, :] - reference[None, :, :], axis=2) ** 2)
        assignment = np.empty(len(X), dtype=int)
        assignment[rows] = cols
        displacement = np.linalg.norm(transformed - reference[assignment], axis=1)
        result = {"rms": float(np.sqrt(np.mean(displacement ** 2))),
                  "max_displacement": float(displacement.max()),
                  "reflected": reflected, "permutation": assignment.tolist()}
        if best is None or (result["rms"], result["max_displacement"]) < (best["rms"], best["max_displacement"]):
            best = result
    return best


def remap_topology(topology: dict, permutation: list[int]) -> dict:
    edges = sorted({tuple(sorted((permutation[a], permutation[b])))
                    for a, b in topology["delaunay_edges"]})
    faces = sorted({tuple(sorted(permutation[a] for a in face))
                    for face in topology["delaunay_faces"]})
    return {
        "delaunay_edges": [list(edge) for edge in edges],
        "delaunay_faces": [list(face) for face in faces],
        "boundary_owner_cycle": canonical_cycle([permutation[site]
                                                   for site in topology["boundary_owner_cycle"]]),
        "counts": topology["counts"],
    }


def classify(run: dict, reference: np.ndarray, reference_H: float,
             reference_topology: dict) -> dict:
    C = np.asarray(run["final"]["centers"], dtype=float)
    comparison = congruence(C, reference)
    H = run["final"]["H"]
    radius = run["final"]["covering_radius"]
    if run["budget"]["stop_reason"] == "max_hole_evals":
        status = "unresolved"
    else:
        covering = H <= 1.0 + STRICT_TOL and radius >= R25 - STRICT_TOL
        if not covering:
            status = "not-covering"
        elif comparison["rms"] <= 1e-5 and comparison["max_displacement"] <= 5e-5:
            status = "reference-like"
        else:
            status = "alternate-numerical-candidate"
    final_topology = run["final"]["topology"]
    remapped = remap_topology(final_topology, comparison["permutation"])
    topology_vs_p0 = {
        "hash_equal": digest(remapped) == reference_topology["hash"],
        "counts_equal": remapped["counts"] == reference_topology["counts"],
        "delaunay_edges_equal": remapped["delaunay_edges"] == reference_topology["delaunay_edges"],
        "delaunay_faces_equal": remapped["delaunay_faces"] == reference_topology["delaunay_faces"],
        "boundary_owner_cycle_equal": remapped["boundary_owner_cycle"] == reference_topology["boundary_owner_cycle"],
    }
    run["comparison"] = {"reference_H": reference_H, **comparison,
                         "covering_gate": {"H_le_1_plus_strict_tol": H <= 1.0 + STRICT_TOL,
                                            "geom_covering_radius_ge_R25": radius >= R25 - STRICT_TOL},
                         "topology_vs_p0": topology_vs_p0,
                         "classification": status}
    return run


def main() -> None:
    reference = centers25()
    started = time.perf_counter()
    reference_records = hole_candidates(reference, R25)
    reference_H = max(r["value"] for r in reference_records)
    reference_q = max(reference_records, key=lambda r: r["value"])["point"]
    reference_topology = topology_signature(reference, reference_records)
    reference_topology["hash"] = digest(reference_topology)
    reference_radius = float(covering_radius(reference))

    runs = []
    sigma_index = 0
    for sigma in SIGMAS:
        for base_seed in SEEDS:
            for restart in range(RESTARTS):
                run = run_one(base_seed, sigma_index, sigma, restart, reference)
                runs.append(classify(run, reference, reference_H, reference_topology))
            sigma_index += 1

    statuses = {status: sum(r["comparison"]["classification"] == status for r in runs)
                for status in ("reference-like", "alternate-numerical-candidate", "not-covering", "unresolved")}
    topology_hash_counts = {}
    active_hash_counts = {}
    active_kind_owners_hash_counts = {}
    topology_change_count = 0
    active_change_count = 0
    active_kind_owners_change_count = 0
    trajectory_count = 0
    for run in runs:
        topology_hash = run["final"]["topology"]["hash"]
        topology_hash_counts[topology_hash] = topology_hash_counts.get(topology_hash, 0) + 1
        previous_topology = None
        previous_active = None
        previous_active_kind_owners = None
        for step in run["trajectory"]:
            trajectory_count += 1
            active_hash_counts[step["active_hash"]] = active_hash_counts.get(step["active_hash"], 0) + 1
            owner_hash = step["active_kind_owners_hash"]
            active_kind_owners_hash_counts[owner_hash] = active_kind_owners_hash_counts.get(owner_hash, 0) + 1
            if previous_topology is not None and step["topology"]["hash"] != previous_topology:
                topology_change_count += 1
            if previous_active is not None and step["active_hash"] != previous_active:
                active_change_count += 1
            if previous_active_kind_owners is not None and owner_hash != previous_active_kind_owners:
                active_kind_owners_change_count += 1
            previous_topology = step["topology"]["hash"]
            previous_active = step["active_hash"]
            previous_active_kind_owners = owner_hash
    topology_fields = ("counts_equal", "delaunay_edges_equal", "delaunay_faces_equal", "boundary_owner_cycle_equal")
    topology_field_matches = {field: sum(r["comparison"]["topology_vs_p0"][field] for r in runs)
                              for field in topology_fields}
    best_congruence_run = min(runs, key=lambda r: (r["comparison"]["rms"], r["comparison"]["max_displacement"]))
    payload = {
        "schema": "n25-p1-smoke-v1",
        "status": "exploration",
        "scope": "P1 fixed-radius unrestricted float search; not Stage B",
        "reference": {"R25": R25, "H": reference_H, "deepest_point": reference_q,
                      "covering_radius": reference_radius, "topology": reference_topology,
                      "centers": reference.tolist()},
        "parameters": {"sigmas": list(SIGMAS), "seeds": list(SEEDS), "restarts": RESTARTS,
                       "rounds": ROUNDS, "n_r": N_R, "max_hole_evals": MAX_HOLE_EVALS,
                       "poll_tries": POLL_TRIES, "active_tol": ACTIVE_TOL, "strict_tol": STRICT_TOL},
        "runs": runs,
        "summary": {"run_count": len(runs), "statuses": statuses,
                     "H_range": [min(r["final"]["H"] for r in runs), max(r["final"]["H"] for r in runs)],
                     "final_topology_hash_counts": topology_hash_counts,
                     "trajectory_active_hash_count": len(active_hash_counts),
                     "trajectory_active_kind_owners_hash_count": len(active_kind_owners_hash_counts),
                     "trajectory_active_hash_counts": active_hash_counts,
                     "trajectory_active_kind_owners_hash_counts": active_kind_owners_hash_counts,
                     "trajectory_count": trajectory_count,
                     "topology_change_count": topology_change_count,
                     "active_hash_change_count": active_change_count,
                     "active_kind_owners_hash_change_count": active_kind_owners_change_count,
                     "reference_topology_hash": reference_topology["hash"],
                     "final_topology_matches_p0": sum(r["comparison"]["topology_vs_p0"]["hash_equal"] for r in runs),
                     "topology_field_matches_p0": topology_field_matches,
                     "best_congruence": {"rms": best_congruence_run["comparison"]["rms"],
                                         "max_displacement": best_congruence_run["comparison"]["max_displacement"],
                                         "run": best_congruence_run["input"]},
                     "alternate_found": statuses["alternate-numerical-candidate"] > 0},
        "reproduction": {"command": ".venv/bin/python src/n25_p1_search.py --smoke",
                         "python": platform.python_version(), "numpy": np.__version__,
                         "scipy": __import__("scipy").__version__},
    }
    manifest_input = {k: v for k, v in payload.items() if k != "runs"}
    manifest_input["runs"] = [{k: v for k, v in run.items() if k != "wall_seconds"} for run in runs]
    payload["deterministic_manifest_hash"] = digest(manifest_input)
    DATA_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    DOC_PATH.write_text(make_report(payload, time.perf_counter() - started))
    print(json.dumps({"output": str(DATA_PATH), "runs": len(runs), "reference_H": reference_H,
                      "H_range": payload["summary"]["H_range"], "statuses": statuses,
                      "best_rms": payload["summary"]["best_congruence"]["rms"],
                      "best_max_displacement": payload["summary"]["best_congruence"]["max_displacement"],
                      "manifest_hash": payload["deterministic_manifest_hash"],
                      "wall_seconds": time.perf_counter() - started}, ensure_ascii=False))


def make_report(payload: dict, elapsed: float) -> str:
    summary = payload["summary"]
    runs = payload["runs"]
    active_counts = {}
    active_kind_owners_counts = {}
    topology_counts = {}
    for run in runs:
        for step in run["trajectory"]:
            active_counts[step["active_hash"]] = active_counts.get(step["active_hash"], 0) + 1
            active_kind_owners_counts[step["active_kind_owners_hash"]] = active_kind_owners_counts.get(step["active_kind_owners_hash"], 0) + 1
            topology_counts[step["topology"]["hash"]] = topology_counts.get(step["topology"]["hash"], 0) + 1
    lines = [
        "# n=25 P1 fixed-radius search",
        "",
        "> 状态：**exploration**。这是无对称浮点发现实验，不是 fixed-radius certificate，也不是 global optimality proof；未进入 Stage B。",
        "",
        "## 复现",
        "",
        "```text",
        ".venv/bin/python src/n25_p1_search.py --smoke",
        ".venv/bin/python -m py_compile src/hole.py src/n25_p1_search.py",
        "```",
        "",
        f"共 {summary['run_count']} runs（四个 sigma、两 seed、每档两 restart），每 run {ROUNDS} 轮、n_r={N_R}、正常路径含最终 best_C 复评估共 38 次 H 评估，硬预算 {MAX_HOLE_EVALS}，bounded poll={POLL_TRIES}。若 stop_reason=max_hole_evals，则分类为 unresolved；wall time 约 {elapsed:.3f}s，且未进入 manifest hash。",
        f"参考构型 H={payload['reference']['H']:.15f}，geom.covering_radius={payload['reference']['covering_radius']:.15f}，参考 topology hash={summary['reference_topology_hash']}。",
        f"最终 H 范围=[{summary['H_range'][0]:.15f}, {summary['H_range'][1]:.15f}]；分类={json.dumps(summary['statuses'], ensure_ascii=False, sort_keys=True)}。best congruence RMS={summary['best_congruence']['rms']:.9g}，max displacement={summary['best_congruence']['max_displacement']:.9g}。",
        "",
        "## 判据与比较",
        "",
        "只有 H <= 1+strict_tol 且独立 geom.covering_radius >= R25-strict_tol 才通过覆盖门；预算耗尽的 run 不论数值门结果如何均分类为 unresolved。其他正常结束 run 再用 rotation/reflection + Hungarian/Procrustes 数值 congruence 分类。topology 比较先按 candidate-site 到 reference-site 的 permutation 重标，再比较规范化组合结构；raw boundary candidate switches 仅是 floating-point fingerprint，不参与 topology hash 或比较。",
        f"alternate-numerical-candidate found: **{summary['alternate_found']}**；本次预算内未发现替代覆盖候选。该结论即使为真也只表示浮点候选，不能表示精确等号覆盖。",
        "",
        "## Active/topology 轨迹",
        "",
        f"轨迹 topology hash 种类数={len(topology_counts)}，跨轮 topology 变化={summary['topology_change_count']}；active 数值 hash 种类数={summary['trajectory_active_hash_count']}，跨轮 active 数值变化={summary['active_hash_change_count']}；active kind/owners hash 种类数={summary['trajectory_active_kind_owners_hash_count']}，跨轮组合变化={summary['active_kind_owners_hash_change_count']}。最终 topology hash 计数={json.dumps(summary['final_topology_hash_counts'], sort_keys=True)}，最终 hash 与 P0 基线相同的 runs={summary['final_topology_matches_p0']}/{summary['run_count']}，字段比较={json.dumps(summary['topology_field_matches_p0'], sort_keys=True)}。每轮完整 active holes、active kind/owners、分离的 active hash、raw fingerprint、topology signature 和 signature_change 均保存在 JSON；候选来源全部标为浮点。",
        "",
        "## 限制",
        "",
        "候选枚举使用浮点 circumcenter、boundary bisector intersection 和 boundary stationary points；geom.covering_radius 是独立数值交叉检查而非区间证明。有限的 12 轮、250 次 H 评估和两次/轮随机 poll 不能排除未搜索的拓扑或连续候选，也不证明全局最优性。",
        "",
        f"确定性 manifest hash（排除 wall time）=`{payload['deterministic_manifest_hash']}`。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
