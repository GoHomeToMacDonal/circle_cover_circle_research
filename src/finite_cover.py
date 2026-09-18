"""Budgeted complete reduction of finite equal-radius disk cover to set cover.

For a finite planar point set Q and radius r, every inclusion-maximal subset
cut out by a radius-r disk is represented by a disk whose centre is either a
point of Q or an intersection of two radius-r circles centred at points of Q.
This module enumerates those locking centres (including the tangent d=2r
case), removes dominated masks, and solves the resulting set-cover problem by
an exhaustive bit-mask branch and bound.

The geometric layer uses floating point and reports its smallest non-locking
membership margin.  A positive-margin feasible witness is independently
checkable.  An infeasibility result is a mathematical certificate only after
all reported incidence decisions have been verified with exact or interval
arithmetic; this module deliberately labels the arithmetic mode ``float``.
Timeout and node/candidate limits always return ``unknown``, never
``infeasible``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class Budget:
    time_limit: float = 60.0
    max_nodes: int = 2_000_000
    max_candidates: int = 2_000_000
    progress_interval: float = 1.0


@dataclass
class MaskFamily:
    status: str
    masks: list[int]
    centers: dict[int, tuple[float, float]]
    candidates: int
    pairs_done: int
    pairs_total: int
    min_nonlocking_margin: float
    elapsed: float
    stop_reason: str | None


@dataclass
class CoverResult:
    status: str
    lower_bound: int
    upper_bound: int
    optimum: int | None
    nodes: int
    elapsed: float
    stop_reason: str | None
    solution_masks: list[int] | None
    solution_centers: list[tuple[float, float]] | None
    arithmetic: str = "float"


class _Monitor:
    def __init__(self, budget: Budget, callback: ProgressCallback | None):
        self.budget = budget
        self.callback = callback
        self.start = time.monotonic()
        self.deadline = self.start + max(0.0, budget.time_limit)
        self.next_report = self.start

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def report(self, stage: str, *, force: bool = False, **fields: object) -> None:
        now = time.monotonic()
        if self.callback is not None and (force or now >= self.next_report):
            self.callback({"stage": stage, "elapsed": now - self.start, **fields})
            self.next_report = now + max(0.01, self.budget.progress_interval)


def _mask(row: np.ndarray) -> int:
    result = 0
    for index in np.flatnonzero(row):
        result |= 1 << int(index)
    return result


def _maximal_with_centers(found: dict[int, tuple[float, float]], n: int,
                          monitor: _Monitor | None = None) -> tuple[list[int], dict[int, tuple[float, float]]] | None:
    ordered = sorted(found, key=lambda item: (-item.bit_count(), item))
    kept: list[int] = []
    by_point: list[list[int]] = [[] for _ in range(n)]
    for ordinal, candidate in enumerate(ordered):
        if monitor is not None and ordinal % 128 == 0 and monitor.expired():
            return None
        bits = [i for i in range(n) if candidate >> i & 1]
        pivot = min(bits, key=lambda i: len(by_point[i]))
        if any(candidate & ~container == 0 for container in by_point[pivot]):
            continue
        kept.append(candidate)
        for i in bits:
            by_point[i].append(candidate)
    return kept, {mask: found[mask] for mask in kept}


def enumerate_masks(points: np.ndarray, radius: float, budget: Budget,
                    callback: ProgressCallback | None = None,
                    geom_tol: float = 2e-12) -> MaskFamily:
    """Completely enumerate maximal radius-r cover masks, subject to budget."""
    monitor = _Monitor(budget, callback)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(points) == 0:
        return MaskFamily("complete", [], {}, 0, 0, 0, math.inf,
                          monitor.elapsed(), None)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    n = len(points)
    total_pairs = n * (n - 1) // 2
    found: dict[int, tuple[float, float]] = {}
    candidates = 0
    pairs_done = 0
    min_margin = math.inf
    radius2 = radius * radius

    def add_center(center: np.ndarray, locking: tuple[int, ...]) -> str | None:
        nonlocal candidates, min_margin
        if monitor.expired():
            return "time_limit"
        if candidates >= budget.max_candidates:
            return "candidate_limit"
        delta = points - center
        d2 = np.einsum("ij,ij->i", delta, delta)
        inside = d2 <= (radius + geom_tol) ** 2
        mask = _mask(inside)
        candidates += 1
        if mask:
            found.setdefault(mask, (float(center[0]), float(center[1])))
        unlocked = np.ones(n, dtype=bool)
        if locking:
            unlocked[list(locking)] = False
        if np.any(unlocked):
            margin = float(np.min(np.abs(d2[unlocked] - radius2)))
            min_margin = min(min_margin, margin)
        return None

    monitor.report("candidate_generation", force=True, pairs_done=0,
                   pairs_total=total_pairs, candidates=0, masks=0)
    for i, point in enumerate(points):
        reason = add_center(point, (i,))
        if reason:
            return MaskFamily("unknown", [], {}, candidates, pairs_done,
                              total_pairs, min_margin, monitor.elapsed(), reason)
    for i in range(n):
        for j in range(i + 1, n):
            if monitor.expired():
                return MaskFamily("unknown", [], {}, candidates, pairs_done,
                                  total_pairs, min_margin, monitor.elapsed(), "time_limit")
            pairs_done += 1
            delta = points[j] - points[i]
            distance2 = float(delta @ delta)
            threshold = (2.0 * radius) ** 2
            if distance2 > threshold + geom_tol * max(1.0, threshold):
                monitor.report("candidate_generation", pairs_done=pairs_done,
                               pairs_total=total_pairs, candidates=candidates,
                               masks=len(found))
                continue
            if distance2 <= geom_tol * geom_tol:
                continue
            distance = math.sqrt(distance2)
            midpoint = 0.5 * (points[i] + points[j])
            height2 = radius2 - 0.25 * distance2
            if height2 < -geom_tol * max(1.0, radius2):
                continue
            height = math.sqrt(max(0.0, height2))
            perpendicular = np.array([-delta[1], delta[0]]) / distance
            signs = (1.0,) if height == 0.0 else (1.0, -1.0)
            for sign in signs:
                reason = add_center(midpoint + sign * height * perpendicular, (i, j))
                if reason:
                    return MaskFamily("unknown", [], {}, candidates, pairs_done,
                                      total_pairs, min_margin, monitor.elapsed(), reason)
            monitor.report("candidate_generation", pairs_done=pairs_done,
                           pairs_total=total_pairs, candidates=candidates,
                           masks=len(found))
    maximal = _maximal_with_centers(found, n, monitor)
    if maximal is None or monitor.expired():
        return MaskFamily("unknown", [], {}, candidates, pairs_done,
                          total_pairs, min_margin, monitor.elapsed(), "time_limit")
    masks, centers = maximal
    full = (1 << n) - 1
    union = 0
    for ordinal, mask in enumerate(masks):
        if ordinal % 128 == 0 and monitor.expired():
            return MaskFamily("unknown", [], {}, candidates, pairs_done,
                              total_pairs, min_margin, monitor.elapsed(), "time_limit")
        union |= mask
    if union != full:
        raise AssertionError("complete candidates failed to cover every input point")
    if monitor.expired():
        return MaskFamily("unknown", [], {}, candidates, pairs_done,
                          total_pairs, min_margin, monitor.elapsed(), "time_limit")
    monitor.report("candidate_generation_done", force=True, pairs_done=pairs_done,
                   pairs_total=total_pairs, candidates=candidates, masks=len(masks),
                   min_nonlocking_margin=min_margin)
    if monitor.expired():
        return MaskFamily("unknown", [], {}, candidates, pairs_done,
                          total_pairs, min_margin, monitor.elapsed(), "time_limit")
    return MaskFamily("complete", masks, centers, candidates, pairs_done,
                      total_pairs, min_margin, monitor.elapsed(), None)


class _SetCover:
    def __init__(self, points: np.ndarray, radius: float, family: MaskFamily,
                 monitor: _Monitor):
        self.points = points
        self.radius = radius
        self.masks = family.masks
        self.centers = family.centers
        self.monitor = monitor
        self.n = len(points)
        self.full = (1 << self.n) - 1
        self.nodes = 0
        self.memo: dict[int, int] = {}
        self.lb_cache: dict[int, int] = {}
        self.stop_reason: str | None = None
        self.by_point: list[list[int]] = [[] for _ in range(self.n)]
        for ordinal, mask in enumerate(self.masks):
            if ordinal % 128 == 0 and self.monitor.expired():
                self.stop_reason = "time_limit"
                return
            bits = mask
            while bits:
                bit = bits & -bits
                self.by_point[bit.bit_length() - 1].append(mask)
                bits ^= bit
        distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        if self.monitor.expired():
            self.stop_reason = "time_limit"
            return
        self.conflicts = []
        for row in distance > 2.0 * radius + 2e-12:
            if len(self.conflicts) % 128 == 0 and self.monitor.expired():
                self.stop_reason = "time_limit"
                return
            self.conflicts.append(_mask(row))

    def greedy(self, uncovered: int | None = None) -> list[int]:
        uncovered = self.full if uncovered is None else uncovered
        answer: list[int] = []
        while uncovered:
            if self.monitor.expired():
                self.stop_reason = "time_limit"
                return answer
            best = 0
            best_size = 0
            for ordinal, mask in enumerate(self.masks):
                if ordinal % 256 == 0 and self.monitor.expired():
                    self.stop_reason = "time_limit"
                    return answer
                size = (mask & uncovered).bit_count()
                if size > best_size:
                    best, best_size = mask, size
            if not best:
                raise AssertionError("uncoverable point")
            answer.append(best)
            uncovered &= ~best
        return answer

    def _packing(self, uncovered: int, high_degree: bool) -> int:
        candidates = uncovered
        count = 0
        while candidates:
            if self.monitor.expired():
                self.stop_reason = "time_limit"
                return 0
            vertices = [i for i in range(self.n) if candidates >> i & 1]
            vertex = (max(vertices, key=lambda i: (self.conflicts[i] & candidates).bit_count())
                      if high_degree else vertices[0])
            count += 1
            candidates &= self.conflicts[vertex]
        return count

    def lower_bound(self, uncovered: int) -> int:
        if not uncovered:
            return 0
        cached = self.lb_cache.get(uncovered)
        if cached is not None:
            return cached
        largest = 0
        for ordinal, mask in enumerate(self.masks):
            if ordinal % 256 == 0 and self.monitor.expired():
                self.stop_reason = "time_limit"
                return 0
            largest = max(largest, (mask & uncovered).bit_count())
        cardinality = math.ceil(uncovered.bit_count() / largest)
        first_packing = self._packing(uncovered, False)
        second_packing = self._packing(uncovered, True)
        if self.stop_reason is not None:
            return 0
        result = max(cardinality, first_packing, second_packing)
        if len(self.lb_cache) < 200_000:
            self.lb_cache[uncovered] = result
        return result

    def _choices(self, uncovered: int) -> list[int]:
        best: list[int] | None = None
        best_key = (math.inf, math.inf)
        bits = uncovered
        while bits:
            if self.monitor.expired():
                self.stop_reason = "time_limit"
                return []
            bit = bits & -bits
            point = bit.bit_length() - 1
            choices = []
            for ordinal, mask in enumerate(self.by_point[point]):
                if ordinal % 256 == 0 and self.monitor.expired():
                    self.stop_reason = "time_limit"
                    return []
                if mask & uncovered:
                    choices.append(mask)
            key = (len(choices), -max((mask & uncovered).bit_count() for mask in choices))
            if key < best_key:
                best, best_key = choices, key
            bits ^= bit
        assert best is not None
        best.sort(key=lambda mask: (-(mask & uncovered).bit_count(), mask))
        reduced: list[int] = []
        residuals: list[int] = []
        for ordinal, mask in enumerate(best):
            if ordinal % 128 == 0 and self.monitor.expired():
                self.stop_reason = "time_limit"
                return []
            residual = mask & uncovered
            if any(residual & ~larger == 0 for larger in residuals):
                continue
            reduced.append(mask)
            residuals.append(residual)
        return reduced

    def _dfs(self, uncovered: int, left: int, target: int) -> list[int] | None:
        if self.monitor.expired():
            self.stop_reason = "time_limit"
            return None
        if self.nodes >= self.monitor.budget.max_nodes:
            self.stop_reason = "node_limit"
            return None
        self.nodes += 1
        self.monitor.report("set_cover_search", nodes=self.nodes, target=target,
                            uncovered=uncovered.bit_count())
        if not uncovered:
            return []
        if left <= 0 or self.lower_bound(uncovered) > left:
            return None
        previous = self.memo.get(uncovered)
        if previous is not None and previous >= left:
            return None
        self.memo[uncovered] = left
        for mask in self._choices(uncovered):
            tail = self._dfs(uncovered & ~mask, left - 1, target)
            if tail is not None:
                return [mask, *tail]
            if self.stop_reason is not None:
                return None
        return None

    def minimum(self, initial_solution: list[int] | None = None) -> CoverResult:
        started = time.monotonic()
        trivial_lower = 0 if not self.full else 1

        def stopped(lower: int, upper: int,
                    solution: list[int] | None = None) -> CoverResult:
            reason = self.stop_reason or "time_limit"
            self.stop_reason = reason
            self.monitor.report("set_cover_stopped", force=True, nodes=self.nodes,
                                lower_bound=lower, upper_bound=upper, reason=reason)
            return CoverResult("unknown", lower, upper, None, self.nodes,
                               time.monotonic() - started, reason, solution,
                               ([self.centers[m] for m in solution]
                                if solution is not None else None))

        if self.stop_reason is not None or self.monitor.expired():
            return stopped(trivial_lower, self.n)
        solution = self.greedy()
        if self.stop_reason is not None:
            return stopped(trivial_lower, self.n)
        if initial_solution is not None:
            if any(mask not in self.centers for mask in initial_solution):
                raise ValueError("initial solution contains a non-candidate mask")
            covered = 0
            for mask in initial_solution:
                covered |= mask
            if covered != self.full:
                raise ValueError("initial solution does not cover all points")
            if len(initial_solution) < len(solution):
                solution = list(initial_solution)
        upper = len(solution)
        lower = self.lower_bound(self.full)
        if self.stop_reason is not None:
            return stopped(trivial_lower, upper, solution)
        self.monitor.report("set_cover_start", force=True, lower_bound=lower,
                            upper_bound=upper, patterns=len(self.masks))
        while lower < upper:
            target = upper - 1
            self.memo.clear()
            candidate = self._dfs(self.full, target, target)
            if self.stop_reason is not None:
                return stopped(lower, upper, solution)
            if candidate is None:
                lower = upper
            else:
                solution = candidate
                upper = len(candidate)
                root_lower = self.lower_bound(self.full)
                if self.stop_reason is not None:
                    return stopped(lower, upper, solution)
                lower = max(lower, root_lower)
                self.monitor.report("set_cover_incumbent", force=True, nodes=self.nodes,
                                    lower_bound=lower, upper_bound=upper)
        if self.monitor.expired():
            return stopped(lower, upper, solution)
        self.monitor.report("set_cover_done", force=True, nodes=self.nodes,
                            optimum=upper)
        if self.monitor.expired():
            return stopped(lower, upper, solution)
        return CoverResult("optimal", upper, upper, upper, self.nodes,
                           time.monotonic() - started, None, solution,
                           [self.centers[m] for m in solution])


def minimum_cover(points: np.ndarray, radius: float, budget: Budget = Budget(),
                  callback: ProgressCallback | None = None,
                  initial_centers: np.ndarray | None = None) -> tuple[MaskFamily, CoverResult]:
    """Return a proven combinatorial optimum, or a bounded ``unknown`` result.

    Optional ``initial_centers`` provide a known cover upper bound.  Each of
    their masks is enlarged to a containing maximal candidate before search.
    """
    overall = _Monitor(budget, callback)
    points = np.asarray(points, dtype=float)
    family = enumerate_masks(points, radius, budget, callback)
    if family.status != "complete":
        result = CoverResult("unknown", 0, len(points), None, 0,
                             overall.elapsed(), family.stop_reason, None, None)
        return family, result
    initial_masks: list[int] | None = None
    if initial_centers is not None:
        initial_masks = []
        for center in np.asarray(initial_centers, dtype=float):
            if overall.expired():
                result = CoverResult("unknown", 0, len(initial_centers), None, 0,
                                     overall.elapsed(), "time_limit", None,
                                     [tuple(map(float, c)) for c in initial_centers])
                return family, result
            distances = np.linalg.norm(points - center, axis=1)
            seed = _mask(distances <= radius + 2e-12)
            containers = [candidate for candidate in family.masks
                          if seed & ~candidate == 0]
            if not containers:
                raise ValueError("initial center mask has no maximal container")
            initial_masks.append(max(containers, key=lambda mask: mask.bit_count()))
        covered = 0
        for mask in initial_masks:
            covered |= mask
        if covered != (1 << len(points)) - 1:
            raise ValueError("initial centers do not cover all input points")
    remaining = max(0.0, budget.time_limit - overall.elapsed())
    solve_budget = Budget(remaining, budget.max_nodes, budget.max_candidates,
                          budget.progress_interval)
    solver = _SetCover(points, radius, family, _Monitor(solve_budget, callback))
    return family, solver.minimum(initial_masks)


def verify_witness(points: np.ndarray, centers: list[tuple[float, float]],
                   radius: float) -> dict[str, float | bool]:
    if not centers and len(points):
        return {"covered": False, "max_distance": math.inf, "slack": -math.inf}
    if not len(points):
        return {"covered": True, "max_distance": 0.0, "slack": radius}
    distances = np.linalg.norm(np.asarray(points)[:, None, :] - np.asarray(centers)[None, :, :], axis=2)
    maximum = float(np.min(distances, axis=1).max())
    return {"covered": maximum <= radius + 2e-12,
            "max_distance": maximum, "slack": radius - maximum}


def _load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload["points"]
    return np.asarray(payload, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--qrho", type=float)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--max-nodes", type=int, default=2_000_000)
    parser.add_argument("--max-candidates", type=int, default=2_000_000)
    parser.add_argument("--progress-interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.qrho is not None:
        from private_region_family import family as qrho_family
        points, *_ = qrho_family(args.qrho)
        radius = args.qrho if args.radius is None else args.radius
    else:
        points = _load_points(args.input)
        if args.radius is None:
            parser.error("--radius is required with --input")
        radius = args.radius

    def report(event: dict[str, object]) -> None:
        print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)

    budget = Budget(args.time_limit, args.max_nodes, args.max_candidates,
                    args.progress_interval)
    family, result = minimum_cover(points, radius, budget, report)
    payload = {
        "points": len(points), "radius": radius,
        "candidate_family": asdict(family), "cover": asdict(result),
        "witness_check": (verify_witness(points, result.solution_centers, radius)
                          if result.solution_centers is not None else None),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
