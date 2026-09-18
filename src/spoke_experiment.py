"""Budgeted finite-cover experiments for concentric radial-layer spokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config25 import R25, centers25
from finite_cover import Budget, minimum_cover, verify_witness
from private_region_family import boundary_with_radial_layers_family

PRESETS: dict[str, tuple[float, ...]] = {
    "none": (),
    "small3": (0.02, 0.05, 0.1),
    "geometric6": (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7),
    "uniform9": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
}


def run_spoke_case(rho: float, level: int, schedule: str,
                   radial_scales: tuple[float, ...], radius_factor: float,
                   budget: Budget, progress_stream=sys.stderr) -> dict[str, object]:
    started = time.monotonic()
    points, owners, names, groups = boundary_with_radial_layers_family(
        rho, level, radial_scales
    )
    radius = rho * radius_factor
    case_name = (f"rho={rho}:level={level}:schedule={schedule}:"
                 f"factor={radius_factor}")

    def progress(event: dict[str, object]) -> None:
        print(json.dumps({"case": case_name, **event}, sort_keys=True),
              file=progress_stream, flush=True)

    owner_centers = centers25() if radius_factor >= 1.0 else None
    known_owner_check = verify_witness(points, list(map(tuple, centers25())), rho)
    family, cover = minimum_cover(points, radius, budget, progress,
                                  initial_centers=owner_centers)
    witness = (verify_witness(points, cover.solution_centers, radius)
               if cover.solution_centers is not None else None)
    mask_payload = "\n".join(
        format(mask, "x") for mask in sorted(family.masks)
    ).encode()
    layer_counts: dict[str, int] = {"outer": 0}
    for name in names:
        if name.startswith("layer:"):
            scale = name.split(":", 2)[1]
            layer_counts[scale] = layer_counts.get(scale, 0) + 1
        else:
            layer_counts["outer"] += 1
    return {
        "rho": rho,
        "level": level,
        "schedule": schedule,
        "radial_scales": list(radial_scales),
        "radius_factor": radius_factor,
        "radius": radius,
        "points": len(points),
        "layer_counts": layer_counts,
        "owner_group_sizes": [len(group) for group in groups],
        "distinct_owner_labels": len(set(owners)),
        "pair_count": len(points) * (len(points) - 1) // 2,
        "known_25_owner_cover": known_owner_check,
        "candidate_family": {
            "status": family.status,
            "candidates": family.candidates,
            "pairs_done": family.pairs_done,
            "pairs_total": family.pairs_total,
            "maximal_masks": len(family.masks),
            "mask_sha256": hashlib.sha256(mask_payload).hexdigest(),
            "min_nonlocking_margin": family.min_nonlocking_margin,
            "elapsed": family.elapsed,
            "stop_reason": family.stop_reason,
        },
        "cover": asdict(cover),
        "witness_check": witness,
        "wall_seconds": time.monotonic() - started,
    }


def run_spoke_experiment(
    rhos: list[float], levels: list[int], schedules: list[str],
    factors: list[float], budget: Budget, output: Path | None,
    progress_stream=sys.stderr,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "description": "concentric homothetic layers with aligned radial spokes",
        "target_disk_radius_original_scale": R25,
        "presets": {name: list(scales) for name, scales in PRESETS.items()},
        "budget_per_case": asdict(budget),
        "cases": [],
    }
    cases: list[dict[str, object]] = payload["cases"]  # type: ignore[assignment]
    for rho in rhos:
        for level in levels:
            for schedule in schedules:
                scales = PRESETS[schedule]
                for factor in factors:
                    start_event = {
                        "stage": "case_start", "rho": rho, "level": level,
                        "schedule": schedule, "radius_factor": factor,
                    }
                    print(json.dumps(start_event, sort_keys=True),
                          file=progress_stream, flush=True)
                    case = run_spoke_case(rho, level, schedule, scales,
                                          factor, budget, progress_stream)
                    cases.append(case)
                    cover = case["cover"]
                    assert isinstance(cover, dict)
                    print(json.dumps({
                        **start_event, "stage": "case_done",
                        "points": case["points"], "status": cover["status"],
                        "lower_bound": cover["lower_bound"],
                        "upper_bound": cover["upper_bound"],
                        "optimum": cover["optimum"], "nodes": cover["nodes"],
                        "wall_seconds": case["wall_seconds"],
                    }, sort_keys=True), file=progress_stream, flush=True)
                    if output is not None:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(json.dumps(payload, indent=2,
                                                     sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.99])
    parser.add_argument("--level", type=int, nargs="+", default=[0])
    parser.add_argument("--schedule", choices=sorted(PRESETS), nargs="+",
                        default=["small3", "geometric6", "uniform9"])
    parser.add_argument("--radius-factor", type=float, nargs="+", default=[0.99])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--progress-interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(level < 0 for level in args.level):
        parser.error("levels must be nonnegative")
    if any(factor <= 0.0 for factor in args.radius_factor):
        parser.error("radius factors must be positive")
    budget = Budget(args.time_limit, args.max_nodes, args.max_candidates,
                    args.progress_interval)
    payload = run_spoke_experiment(
        args.rho, args.level, args.schedule, args.radius_factor,
        budget, args.output,
    )
    summaries = []
    for case in payload["cases"]:
        cover = case["cover"]
        summaries.append({key: case[key] for key in
                          ("rho", "level", "schedule", "radius_factor",
                           "points")} |
                         {key: cover[key] for key in
                          ("status", "lower_bound", "upper_bound", "optimum",
                           "nodes")})
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
