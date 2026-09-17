"""Exact finite partition search for the 25-circle covering problem.

For a finite point set P in the unit disk, enumerate every maximal subset that
can be covered by one disk of radius

    lambda = 1 / (sqrt(6) + sqrt(3)).

The partition problem is then an ordinary set-cover decision problem.  Cover
and partition have the same optimum: from a cover, assign every point to one of
the disks containing it.

This module is an exploratory floating-point searcher.  It records geometric
margins and deterministic hashes, but a final mathematical certificate still
needs exact algebraic/interval verification of all membership decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

LAMBDA = 1.0 / (math.sqrt(6.0) + math.sqrt(3.0))
GEOM_TOL = 2e-12


@dataclass
class SolveResult:
    status: str
    target: int
    lower_bound: int
    greedy_upper_bound: int
    nodes: int
    elapsed: float
    solution_masks: list[int] | None


@dataclass
class CandidateResult:
    seed: int
    points: int
    patterns: int
    generation_seconds: float
    min_membership_margin: float
    solve: SolveResult
    point_hash: str
    pattern_hash: str


def _dedup_points(points: np.ndarray, digits: int = 13) -> np.ndarray:
    """Deterministically remove numerical duplicates while preserving order."""
    seen: set[tuple[float, float]] = set()
    out: list[np.ndarray] = []
    for p in points:
        key = (round(float(p[0]), digits), round(float(p[1]), digits))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return np.asarray(out, dtype=float)


def generate_candidate(seed: int) -> np.ndarray:
    """Generate one reproducible dense D8-symmetric polar point set.

    The first 100-candidate round showed that 5--7 rings with spacing near 0.18
    were usually coverable greedily by at most 24 disks.  This second family
    therefore uses 8--10 radial rings and angular spacing near 0.10--0.135.
    The origin and a complete outer boundary ring are always present.
    """
    rng = np.random.default_rng(seed)
    n_rings = 8 + seed % 3
    spacing = rng.uniform(0.100, 0.135)
    radii = np.linspace(1.0 / n_rings, 1.0, n_rings)
    if n_rings > 1:
        jitter = rng.uniform(-0.14, 0.14, n_rings - 1) / n_rings
        radii[:-1] = np.clip(radii[:-1] + jitter, 0.055, 0.975)
        radii[:-1].sort()
    # Include the three nontrivial radii at which the known 25-circle cover
    # changes its active pair of rings.  Replacing nearest grid radii retains
    # density while making candidates sensitive to the exact construction.
    critical = np.array([
        LAMBDA,
        LAMBDA * (1.0 + math.sqrt(2.0)),
        LAMBDA * math.sqrt(5.0 + 2.0 * math.sqrt(2.0)),
    ])
    for value in critical:
        index = int(np.argmin(np.abs(radii[:-1] - value)))
        radii[index] = value
    radii[:-1].sort()
    radii[-1] = 1.0

    points = [np.array([0.0, 0.0])]
    for j, radius in enumerate(radii):
        ideal = max(8, math.ceil(2.0 * math.pi * radius / spacing))
        count = 8 * math.ceil(ideal / 8)
        # phase is an integer multiple of pi/count, so reflection in the x-axis
        # and rotation by pi/4 preserve every ring.
        phase = (rng.integers(0, 2) * math.pi / count)
        if (seed + j) % 4 == 0:
            phase += math.pi / 8.0
        angles = phase + np.arange(count) * (2.0 * math.pi / count)
        points.extend(np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]))
    result = _dedup_points(np.asarray(points))
    if np.max(np.linalg.norm(result, axis=1)) > 1.0 + 1e-12:
        raise AssertionError("candidate point escaped the unit disk")
    return result


def _mask_from_bool(row: np.ndarray) -> int:
    mask = 0
    for i in np.flatnonzero(row):
        mask |= 1 << int(i)
    return mask


def locking_centers(points: np.ndarray, radius: float = LAMBDA) -> np.ndarray:
    """Complete locking-centre list for maximal radius-r subsets of points."""
    n = len(points)
    chunks = [points.copy()]
    i, j = np.triu_indices(n, 1)
    a = points[i]
    delta = points[j] - a
    distance = np.linalg.norm(delta, axis=1)
    valid = (distance > GEOM_TOL) & (distance <= 2.0 * radius + GEOM_TOL)
    a, delta, distance = a[valid], delta[valid], distance[valid]
    if len(distance):
        midpoint = a + 0.5 * delta
        height = np.sqrt(np.maximum(0.0, radius * radius - 0.25 * distance * distance))
        perpendicular = np.column_stack([-delta[:, 1], delta[:, 0]]) / distance[:, None]
        chunks.extend([midpoint + height[:, None] * perpendicular,
                       midpoint - height[:, None] * perpendicular])
    return np.concatenate(chunks)


def _maximal_masks(masks: Iterable[int], n_points: int) -> list[int]:
    """Drop every mask contained in another mask."""
    ordered = sorted(set(masks), key=lambda m: (-m.bit_count(), m))
    kept: list[int] = []
    by_point: list[list[int]] = [[] for _ in range(n_points)]
    for mask in ordered:
        bits = [i for i in range(n_points) if (mask >> i) & 1]
        if not bits:
            continue
        pivot = min(bits, key=lambda i: len(by_point[i]))
        if any(mask & ~container == 0 for container in by_point[pivot]):
            continue
        idx = len(kept)
        kept.append(mask)
        for i in bits:
            by_point[i].append(kept[idx])
    return kept


def candidate_masks(points: np.ndarray, radius: float = LAMBDA,
                    chunk_size: int = 2048) -> tuple[list[int], float]:
    """Enumerate all distinct maximal subsets coverable by a radius-r disk.

    Returns the masks and the smallest absolute point-to-boundary margin seen at
    a locking centre.  A tiny margin flags instances needing exact re-checking.
    """
    centers = locking_centers(points, radius)
    masks: list[int] = []
    min_margin = math.inf
    for start in range(0, len(centers), chunk_size):
        block = centers[start:start + chunk_size]
        distances = np.linalg.norm(block[:, None, :] - points[None, :, :], axis=2)
        min_margin = min(min_margin, float(np.min(np.abs(distances - radius))))
        inside = distances <= radius + GEOM_TOL
        masks.extend(_mask_from_bool(row) for row in inside if row.any())
    maximal = _maximal_masks(masks, len(points))
    covered = 0
    for mask in maximal:
        covered |= mask
    if covered != (1 << len(points)) - 1:
        raise AssertionError("candidate masks do not cover every point")
    return maximal, min_margin


def masks_hash(masks: Iterable[int]) -> str:
    payload = "\n".join(format(mask, "x") for mask in sorted(masks)).encode()
    return hashlib.sha256(payload).hexdigest()


def points_hash(points: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()


class PartitionSolver:
    """Bitset branch-and-bound solver for coverability by at most k groups."""

    def __init__(self, points: np.ndarray, masks: list[int], radius: float = LAMBDA):
        self.points = points
        self.masks = masks
        self.radius = radius
        self.n = len(points)
        self.full = (1 << self.n) - 1
        self.by_point: list[list[int]] = [[] for _ in range(self.n)]
        for mask in masks:
            for i in range(self.n):
                if (mask >> i) & 1:
                    self.by_point[i].append(mask)
        distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        self.conflict = []
        for i in range(self.n):
            row = distance[i] > 2.0 * radius + GEOM_TOL
            self.conflict.append(_mask_from_bool(row))
        self.nodes = 0
        self.deadline = math.inf
        self.max_nodes = 0
        self.memo: dict[int, int] = {}
        self.lb_cache: dict[int, int] = {}
        self.timed_out = False

    def greedy_cover(self, uncovered: int | None = None) -> list[int]:
        uncovered = self.full if uncovered is None else uncovered
        answer: list[int] = []
        while uncovered:
            best = max(self.masks, key=lambda m: (m & uncovered).bit_count())
            if not (best & uncovered):
                raise AssertionError("uncoverable point")
            answer.append(best)
            uncovered &= ~best
        return answer

    def _packing_bound_once(self, uncovered: int, prefer_degree: bool) -> int:
        candidates = uncovered
        size = 0
        while candidates:
            if prefer_degree:
                vertices = [i for i in range(self.n) if (candidates >> i) & 1]
                vertex = max(vertices, key=lambda i: (self.conflict[i] & candidates).bit_count())
            else:
                vertex = (candidates & -candidates).bit_length() - 1
            size += 1
            candidates &= self.conflict[vertex]
        return size

    def lower_bound(self, uncovered: int) -> int:
        cached = self.lb_cache.get(uncovered)
        if cached is not None:
            return cached
        if not uncovered:
            return 0
        maximum = max((mask & uncovered).bit_count() for mask in self.masks)
        cardinality = math.ceil(uncovered.bit_count() / maximum)
        packing = max(self._packing_bound_once(uncovered, False),
                      self._packing_bound_once(uncovered, True))
        result = max(cardinality, packing)
        if len(self.lb_cache) < 200_000:
            self.lb_cache[uncovered] = result
        return result

    def _choose_point(self, uncovered: int) -> tuple[int, list[int]]:
        best_point = -1
        best_choices: list[int] = []
        best_key = (math.inf, math.inf)
        bits = uncovered
        while bits:
            low = bits & -bits
            point = low.bit_length() - 1
            choices = [m for m in self.by_point[point] if m & uncovered]
            # Prefer a constrained point; break ties by larger best coverage.
            key = (len(choices), -max((m & uncovered).bit_count() for m in choices))
            if key < best_key:
                best_point, best_choices, best_key = point, choices, key
            bits ^= low
        # State-dependent dominance among branches containing the chosen point.
        best_choices.sort(key=lambda m: (-(m & uncovered).bit_count(), m))
        reduced: list[int] = []
        residuals: list[int] = []
        for mask in best_choices:
            residual = mask & uncovered
            if any(residual & ~larger == 0 for larger in residuals):
                continue
            reduced.append(mask)
            residuals.append(residual)
        return best_point, reduced

    def _dfs(self, uncovered: int, groups_left: int) -> list[int] | None:
        self.nodes += 1
        if self.nodes > self.max_nodes or time.monotonic() > self.deadline:
            self.timed_out = True
            return None
        if not uncovered:
            return []
        if groups_left <= 0 or self.lower_bound(uncovered) > groups_left:
            return None
        previous = self.memo.get(uncovered)
        if previous is not None and previous >= groups_left:
            return None
        self.memo[uncovered] = groups_left
        _, choices = self._choose_point(uncovered)
        for mask in choices:
            tail = self._dfs(uncovered & ~mask, groups_left - 1)
            if tail is not None:
                return [mask, *tail]
            if self.timed_out:
                return None
        return None

    def decide(self, target: int = 24, time_limit: float = 1.0,
               max_nodes: int = 100_000) -> SolveResult:
        start = time.monotonic()
        greedy = self.greedy_cover()
        lower = self.lower_bound(self.full)
        if len(greedy) <= target:
            return SolveResult("feasible", target, lower, len(greedy), 0,
                               time.monotonic() - start, greedy)
        if lower > target:
            return SolveResult("infeasible", target, lower, len(greedy), 0,
                               time.monotonic() - start, None)
        self.nodes = 0
        self.max_nodes = max_nodes
        self.deadline = start + time_limit
        self.memo.clear()
        self.timed_out = False
        solution = self._dfs(self.full, target)
        if solution is not None:
            status = "feasible"
        elif self.timed_out:
            status = "unknown"
        else:
            status = "infeasible"
        return SolveResult(status, target, lower, len(greedy), self.nodes,
                           time.monotonic() - start, solution)


def evaluate_seed(seed: int, target: int, time_limit: float,
                  max_nodes: int) -> CandidateResult:
    points = generate_candidate(seed)
    start = time.monotonic()
    masks, margin = candidate_masks(points)
    generation = time.monotonic() - start
    solve = PartitionSolver(points, masks).decide(target, time_limit, max_nodes)
    return CandidateResult(seed, len(points), len(masks), generation, margin,
                           solve, points_hash(points), masks_hash(masks))


def run_batch(count: int, start_seed: int, target: int, time_limit: float,
              max_nodes: int, output: Path | None) -> list[CandidateResult]:
    results = []
    for seed in range(start_seed, start_seed + count):
        result = evaluate_seed(seed, target, time_limit, max_nodes)
        results.append(result)
        s = result.solve
        print(f"seed={seed:3d} points={result.points:3d} patterns={result.patterns:5d} "
              f"gen={result.generation_seconds:6.3f}s lb={s.lower_bound:2d} "
              f"greedy={s.greedy_upper_bound:2d} status={s.status:10s} "
              f"nodes={s.nodes:7d} solve={s.elapsed:6.3f}s", flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lambda": LAMBDA,
            "target": target,
            "count": count,
            "start_seed": start_seed,
            "results": [asdict(r) for r in results],
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--target", type=int, default=24)
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run_batch(args.count, args.start_seed, args.target,
                        args.time_limit, args.max_nodes, args.output)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.solve.status] = counts.get(result.solve.status, 0) + 1
    best = max(results, key=lambda r: (r.solve.lower_bound,
                                       r.solve.greedy_upper_bound,
                                       -r.solve.elapsed))
    print("summary", counts)
    print(f"best seed={best.seed} lower={best.solve.lower_bound} "
          f"greedy={best.solve.greedy_upper_bound} points={best.points} "
          f"patterns={best.patterns}")


if __name__ == "__main__":
    main()
