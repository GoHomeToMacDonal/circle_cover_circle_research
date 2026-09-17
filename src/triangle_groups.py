"""Search 25 acute-isosceles three-point locking groups.

Each of the known 25 radius-LAMBDA covering disks receives three boundary
points.  Relative to a disk's outward radial axis, the boundary normals are
{-beta, +beta, pi}.  For 0 < beta < pi/2 the resulting isosceles triangle is
acute, so its unique minimum enclosing circle is exactly the chosen disk.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from certificate import mec_center
from partition_search import LAMBDA, PartitionSolver, candidate_masks, masks_hash, points_hash

SQRT2 = math.sqrt(2.0)
A = math.sqrt(2.0 + SQRT2)
ETA = math.sqrt(2.0 - SQRT2)


@dataclass(frozen=True)
class TriangleParameters:
    beta_o: float
    beta_a: float
    beta_b: float
    beta_c: float
    phase_o: float

    def degrees(self) -> dict[str, float]:
        return {key: math.degrees(value) for key, value in asdict(self).items()}


@dataclass
class TrialResult:
    seed: int
    parameters_degrees: dict[str, float]
    point_count: int
    pattern_count: int
    optimum: float | None
    dual_bound: float | None
    status: int
    solve_seconds: float
    point_hash: str
    pattern_hash: str


def known_centers() -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return normalized centres, labels, and outward-axis angles."""
    centers = [np.zeros(2)]
    labels = ["O"]
    axes = [0.0]
    for name, rho, offset in (
        ("A", A, math.pi / 8.0),
        ("B", 2.0 + SQRT2, 0.0),
        ("C", 2.0 * A, math.pi / 8.0),
    ):
        for k in range(8):
            angle = offset + k * math.pi / 4.0
            centers.append(LAMBDA * rho * np.array([math.cos(angle), math.sin(angle)]))
            labels.append(f"{name}{k}")
            axes.append(angle)
    return np.asarray(centers), labels, np.asarray(axes)


def trial_parameters(seed: int) -> TriangleParameters:
    """One deterministic parameter choice from the admissible open ranges."""
    rng = np.random.default_rng(seed)
    return TriangleParameters(
        beta_o=math.radians(rng.uniform(42.0, 87.0)),
        beta_a=math.radians(rng.uniform(35.0, 87.0)),
        beta_b=math.radians(rng.uniform(45.5, 88.5)),
        beta_c=math.radians(rng.uniform(68.0, 88.5)),
        phase_o=math.radians(rng.uniform(0.0, 45.0)),
    )


def triangle_points(parameters: TriangleParameters) -> tuple[np.ndarray, list[str], list[list[int]]]:
    centers, circle_labels, axes = known_centers()
    points: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[list[int]] = []
    for i, (center, circle, axis) in enumerate(zip(centers, circle_labels, axes)):
        if circle == "O":
            beta = parameters.beta_o
            axis = parameters.phase_o
        elif circle.startswith("A"):
            beta = parameters.beta_a
        elif circle.startswith("B"):
            beta = parameters.beta_b
        else:
            beta = parameters.beta_c
        group = []
        for suffix, relative in (("-", -beta), ("+", beta), ("i", math.pi)):
            angle = axis + relative
            point = center + LAMBDA * np.array([math.cos(angle), math.sin(angle)])
            group.append(len(points))
            points.append(point)
            labels.append(f"{circle}{suffix}")
        groups.append(group)
    return np.asarray(points), labels, groups


def validate_locking_groups(parameters: TriangleParameters,
                            atol: float = 2e-10) -> dict[str, float]:
    points, _, groups = triangle_points(parameters)
    centers, _, _ = known_centers()
    if len(points) != 75:
        raise AssertionError("expected 75 points")
    if np.linalg.norm(points, axis=1).max() > 1.0 + atol:
        raise AssertionError("triangle point outside target unit disk")
    min_acute_margin = math.inf
    max_radius_error = 0.0
    max_center_error = 0.0
    for center, group in zip(centers, groups):
        triangle = points[group]
        side2 = sorted(float(x) for x in (
            np.sum((triangle[0] - triangle[1]) ** 2),
            np.sum((triangle[0] - triangle[2]) ** 2),
            np.sum((triangle[1] - triangle[2]) ** 2),
        ))
        min_acute_margin = min(min_acute_margin, side2[0] + side2[1] - side2[2])
        fitted = mec_center(triangle)
        radius = float(np.linalg.norm(triangle - fitted, axis=1).max())
        max_radius_error = max(max_radius_error, abs(radius - LAMBDA))
        max_center_error = max(max_center_error, float(np.linalg.norm(fitted - center)))
    if min_acute_margin <= 1e-12:
        raise AssertionError("non-acute triangle")
    if max_radius_error > atol or max_center_error > atol:
        raise AssertionError("MEC does not recover the intended circle")
    return {
        "min_acute_margin": min_acute_margin,
        "max_radius_error": max_radius_error,
        "max_center_error": max_center_error,
        "max_point_radius": float(np.linalg.norm(points, axis=1).max()),
    }


def set_cover_milp(point_count: int, masks: list[int],
                   time_limit: float = 30.0):
    rows: list[int] = []
    cols: list[int] = []
    for j, original in enumerate(masks):
        mask = original
        while mask:
            bit = mask & -mask
            rows.append(bit.bit_length() - 1)
            cols.append(j)
            mask ^= bit
    matrix = csc_matrix((np.ones(len(rows)), (rows, cols)),
                        shape=(point_count, len(masks)))
    return milp(
        c=np.ones(len(masks)),
        constraints=LinearConstraint(matrix, 1.0, np.inf),
        integrality=np.ones(len(masks)),
        bounds=Bounds(0.0, 1.0),
        options={"time_limit": time_limit, "presolve": True},
    )


def evaluate(seed: int, time_limit: float = 30.0) -> tuple[TrialResult, np.ndarray, list[int]]:
    parameters = trial_parameters(seed)
    validate_locking_groups(parameters)
    points, _, _ = triangle_points(parameters)
    masks, _ = candidate_masks(points)
    start = time.monotonic()
    result = set_cover_milp(len(points), masks, time_limit)
    elapsed = time.monotonic() - start
    trial = TrialResult(
        seed=seed,
        parameters_degrees=parameters.degrees(),
        point_count=len(points),
        pattern_count=len(masks),
        optimum=float(result.fun) if result.fun is not None else None,
        dual_bound=float(result.mip_dual_bound) if result.mip_dual_bound is not None else None,
        status=int(result.status),
        solve_seconds=elapsed,
        point_hash=points_hash(points),
        pattern_hash=masks_hash(masks),
    )
    return trial, points, masks


def run_trials(count: int, start_seed: int, time_limit: float,
               output: Path | None, best_points: Path | None) -> list[TrialResult]:
    trials: list[TrialResult] = []
    best: tuple[tuple[float, float], TrialResult, np.ndarray] | None = None
    for seed in range(start_seed, start_seed + count):
        trial, points, _ = evaluate(seed, time_limit)
        trials.append(trial)
        score = (trial.dual_bound if trial.dual_bound is not None else -1.0,
                 trial.optimum if trial.optimum is not None else -1.0)
        if best is None or score > best[0]:
            best = (score, trial, points.copy())
        print(f"seed={seed:3d} patterns={trial.pattern_count:4d} "
              f"opt={trial.optimum!s:>6} dual={trial.dual_bound!s:>6} "
              f"status={trial.status} sec={trial.solve_seconds:.3f}", flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "lambda": LAMBDA,
            "count": count,
            "start_seed": start_seed,
            "trials": [asdict(trial) for trial in trials],
        }, indent=2, sort_keys=True) + "\n")
    if best_points is not None and best is not None:
        best_points.parent.mkdir(parents=True, exist_ok=True)
        np.save(best_points, best[2])
        metadata = best_points.with_suffix(".json")
        metadata.write_text(json.dumps(asdict(best[1]), indent=2, sort_keys=True) + "\n")
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--output", type=Path,
                        default=Path("data/triangle_trials.json"))
    parser.add_argument("--best-points", type=Path,
                        default=Path("data/triangle_best_points.npy"))
    args = parser.parse_args()
    trials = run_trials(args.count, args.start_seed, args.time_limit,
                        args.output, args.best_points)
    best = max(trials, key=lambda trial: (
        trial.dual_bound if trial.dual_bound is not None else -1.0,
        trial.optimum if trial.optimum is not None else -1.0,
    ))
    print("best", json.dumps(asdict(best), sort_keys=True))


if __name__ == "__main__":
    main()
