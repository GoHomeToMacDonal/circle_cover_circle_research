"""Adaptive refinement: add guards at gaps left on continuous radial spokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config25 import LABELS, centers25
from finite_cover import Budget, minimum_cover, verify_witness
from private_region_family import (
    boundary_dense_family,
    boundary_with_radial_layers_family,
)


def continuous_spoke_gaps(rho: float, level: int,
                          witness_centers: np.ndarray, radius: float,
                          tolerance: float = 1e-10) -> list[dict[str, object]]:
    """Return uncovered parameter intervals on owner-centre-to-outer segments."""
    outer, owners, names, _ = boundary_dense_family(rho, level)
    center_by_owner = dict(zip(LABELS, centers25()))
    gaps: list[dict[str, object]] = []
    checked_radius = radius + 2e-12
    for outer_index, (point, owner, name) in enumerate(zip(outer, owners, names)):
        center = center_by_owner[owner]
        direction = point - center
        quadratic = float(direction @ direction)
        intervals: list[tuple[float, float]] = []
        for witness in np.asarray(witness_centers, dtype=float):
            delta = center - witness
            linear = 2.0 * float(delta @ direction)
            constant = float(delta @ delta) - checked_radius * checked_radius
            discriminant = linear * linear - 4.0 * quadratic * constant
            if discriminant < 0.0:
                continue
            root = math.sqrt(max(0.0, discriminant))
            lo = max(0.0, (-linear - root) / (2.0 * quadratic))
            hi = min(1.0, (-linear + root) / (2.0 * quadratic))
            if lo <= hi:
                intervals.append((lo, hi))
        intervals.sort()
        reach = 0.0
        spoke_gaps: list[tuple[float, float]] = []
        for lo, hi in intervals:
            if hi <= reach:
                continue
            if lo > reach + tolerance:
                spoke_gaps.append((reach, lo))
            reach = max(reach, hi)
        if reach < 1.0 - tolerance:
            spoke_gaps.append((reach, 1.0))
        for lo, hi in spoke_gaps:
            gaps.append({
                "outer_index": outer_index,
                "owner": owner,
                "outer_name": name,
                "lo": lo,
                "hi": hi,
                "midpoint": 0.5 * (lo + hi),
                "width": hi - lo,
            })
    return gaps


def refined_points(source_case: dict[str, object]) -> tuple[np.ndarray, list[dict[str, object]]]:
    rho = float(source_case["rho"])
    level = int(source_case["level"])
    scales = tuple(float(value) for value in source_case["radial_scales"])
    cover = source_case["cover"]
    assert isinstance(cover, dict)
    witness = np.asarray(cover["solution_centers"], dtype=float)
    radius = float(source_case["radius"])
    points, *_ = boundary_with_radial_layers_family(rho, level, scales)
    outer, owners, _, _ = boundary_dense_family(rho, level)
    center_by_owner = dict(zip(LABELS, centers25()))
    gaps = continuous_spoke_gaps(rho, level, witness, radius)
    guards = []
    for gap in gaps:
        outer_index = int(gap["outer_index"])
        center = center_by_owner[owners[outer_index]]
        guards.append(center + float(gap["midpoint"]) *
                      (outer[outer_index] - center))
    if guards:
        points = np.concatenate([points, np.asarray(guards)], axis=0)
    return points, gaps


def run_refinement(source: Path, budget: Budget, output: Path | None,
                   progress_stream=sys.stderr) -> dict[str, object]:
    source_payload = json.loads(source.read_text())
    source_case = source_payload["cases"][0]
    points, gaps = refined_points(source_case)
    radius = float(source_case["radius"])
    started = time.monotonic()

    def progress(event: dict[str, object]) -> None:
        print(json.dumps({"case": "adaptive_spoke_refinement", **event},
                         sort_keys=True), file=progress_stream, flush=True)

    family, cover = minimum_cover(points, radius, budget, progress)
    witness = (verify_witness(points, cover.solution_centers, radius)
               if cover.solution_centers is not None else None)
    result = {
        "description": "midpoint guards for continuous-spoke gaps",
        "source": str(source),
        "source_case": {key: source_case[key] for key in
                        ("rho", "level", "schedule", "radial_scales",
                         "radius_factor", "radius", "points")},
        "budget": asdict(budget),
        "gaps": gaps,
        "guard_count": len(gaps),
        "points": len(points),
        "candidate_family": {
            "status": family.status,
            "candidates": family.candidates,
            "pairs_done": family.pairs_done,
            "pairs_total": family.pairs_total,
            "maximal_masks": len(family.masks),
            "mask_sha256": hashlib.sha256(
                "\n".join(format(mask, "x") for mask in sorted(family.masks)).encode()
            ).hexdigest(),
            "min_nonlocking_margin": family.min_nonlocking_margin,
            "elapsed": family.elapsed,
            "stop_reason": family.stop_reason,
        },
        "cover": asdict(cover),
        "witness_check": witness,
        "wall_seconds": time.monotonic() - started,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--time-limit", type=float, default=90.0)
    parser.add_argument("--max-nodes", type=int, default=500_000)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_refinement(
        args.source,
        Budget(args.time_limit, args.max_nodes, args.max_candidates,
               args.progress_interval),
        args.output,
    )
    print(json.dumps({
        "guard_count": result["guard_count"], "points": result["points"],
        "cover": result["cover"], "witness_check": result["witness_check"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
