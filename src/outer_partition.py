"""Finite verifier for the outer-ring 16-disk subproblem.

The 56-point set consists of:
* the 16 intersections of the known outer disks with the target unit circle;
* 24 equally spaced boundary points, at angles 7.5 + 15*j degrees;
* the radially innermost boundary point of each of the 16 outer disks.

Any subset coverable by a radius-LAMBDA disk is a clique in the compatibility
graph (two points are adjacent when their distance is at most 2*LAMBDA).  The
verifier enumerates all maximal cliques and proves that even this relaxation
cannot cover the 56 vertices with 15 cliques.  It also emits an explicit
16-group partition covered by the known B/C disks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from partition_search import LAMBDA, PartitionSolver, masks_hash, points_hash

SQRT2 = math.sqrt(2.0)
A = math.sqrt(2.0 + SQRT2)
ETA = math.sqrt(2.0 - SQRT2)


def rotate(k: int, point: tuple[float, float]) -> np.ndarray:
    angle = k * math.pi / 4.0
    c, s = math.cos(angle), math.sin(angle)
    x, y = point
    return np.array([c * x - s * y, s * x + c * y])


def outer_points() -> tuple[np.ndarray, list[str]]:
    points: list[np.ndarray] = []
    labels: list[str] = []

    for name, base in (
        ("X-", (2.0 + 1.5 * SQRT2, 0.5 * SQRT2)),
        ("X+", (2.0 + SQRT2, 1.0 + SQRT2)),
    ):
        for k in range(8):
            points.append(LAMBDA * rotate(k, base))
            labels.append(f"{name}{k}")

    for j in range(24):
        angle = (j + 0.5) * math.pi / 12.0
        points.append(np.array([math.cos(angle), math.sin(angle)]))
        labels.append(f"Q{j}")

    for k in range(8):
        points.append(LAMBDA * rotate(k, (1.0 + SQRT2, 0.0)))
        labels.append(f"YB{k}")

    c_inner = (2.0 + SQRT2 - 0.5 * A, SQRT2 - 0.5 * ETA)
    for k in range(8):
        points.append(LAMBDA * rotate(k, c_inner))
        labels.append(f"YC{k}")

    return np.asarray(points), labels


def compatibility_graph(points: np.ndarray) -> list[int]:
    """Closed-neighbour-free adjacency masks for distance <= 2*LAMBDA."""
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    adjacency = []
    for i in range(len(points)):
        row = distance[i] <= 2.0 * LAMBDA + 2e-12
        row[i] = False
        mask = 0
        for j in np.flatnonzero(row):
            mask |= 1 << int(j)
        adjacency.append(mask)
    return adjacency


def maximal_cliques(adjacency: list[int]) -> list[int]:
    """Enumerate all maximal cliques with deterministic Bron--Kerbosch pivoting."""
    n = len(adjacency)
    answer: list[int] = []

    def visit(clique: int, candidates: int, excluded: int) -> None:
        if not candidates and not excluded:
            answer.append(clique)
            return
        union = candidates | excluded
        if union:
            vertices = [i for i in range(n) if (union >> i) & 1]
            pivot = max(vertices,
                        key=lambda i: ((candidates & adjacency[i]).bit_count(), -i))
            branch = candidates & ~adjacency[pivot]
        else:
            branch = candidates
        while branch:
            bit = branch & -branch
            vertex = bit.bit_length() - 1
            visit(clique | bit,
                  candidates & adjacency[vertex],
                  excluded & adjacency[vertex])
            candidates ^= bit
            excluded |= bit
            branch ^= bit

    visit(0, (1 << n) - 1, 0)
    return sorted(answer, key=lambda mask: (-mask.bit_count(), mask))


def known_centers() -> tuple[list[np.ndarray], list[str]]:
    centers: list[np.ndarray] = []
    labels: list[str] = []
    for name, base in (("B", (2.0 + SQRT2, 0.0)),
                       ("C", (2.0 + SQRT2, SQRT2))):
        for k in range(8):
            centers.append(LAMBDA * rotate(k, base))
            labels.append(f"{name}{k}")
    return centers, labels


def explicit_partition() -> list[list[int]]:
    """A disjoint 16-group cover, with B_k group followed by C_k group."""
    groups: list[list[int]] = []
    for k in range(8):
        groups.append([
            k,                              # X-_k
            8 + (k - 1) % 8,               # X+_{k-1}
            16 + (3 * k - 1) % 24,         # Q_{3k-1}
            16 + (3 * k) % 24,             # Q_{3k}
            40 + k,                         # YB_k
        ])
    for k in range(8):
        groups.append([
            16 + (3 * k + 1) % 24,         # Q_{3k+1}
            48 + k,                         # YC_k
        ])
    return groups


def graph_hash(adjacency: list[int]) -> str:
    payload = "\n".join(format(mask, "x") for mask in adjacency).encode()
    return hashlib.sha256(payload).hexdigest()


def verify(output: Path | None = None) -> dict:
    points, labels = outer_points()
    adjacency = compatibility_graph(points)
    cliques = maximal_cliques(adjacency)

    if len(points) != 56:
        raise AssertionError(f"expected 56 points, got {len(points)}")
    sizes = {size: sum(mask.bit_count() == size for mask in cliques)
             for size in sorted({mask.bit_count() for mask in cliques})}
    if sizes != {3: 8, 4: 40, 5: 8}:
        raise AssertionError(f"unexpected maximal-clique sizes: {sizes}")

    solver = PartitionSolver(points, cliques)
    lower = solver.decide(target=15, time_limit=300.0, max_nodes=10_000_000)
    if lower.status != "infeasible":
        raise AssertionError(f"15-group search did not prove infeasibility: {lower}")

    groups = explicit_partition()
    flat = sorted(i for group in groups for i in group)
    if flat != list(range(56)):
        raise AssertionError("explicit groups are not a partition")
    centers, center_labels = known_centers()
    for group, center in zip(groups, centers):
        distance = np.linalg.norm(points[group] - center, axis=1)
        if np.max(distance) > LAMBDA + 2e-12:
            raise AssertionError("explicit group is not covered by its known disk")

    upper = PartitionSolver(points, cliques).decide(target=16)
    if upper.status != "feasible":
        raise AssertionError("maximal-clique model unexpectedly needs more than 16")

    certificate = {
        "lambda": LAMBDA,
        "point_count": len(points),
        "point_hash": points_hash(points),
        "graph_hash": graph_hash(adjacency),
        "maximal_clique_count": len(cliques),
        "maximal_clique_sizes": sizes,
        "maximal_clique_hash": masks_hash(cliques),
        "search_15": {
            "status": lower.status,
            "root_lower_bound": lower.lower_bound,
            "greedy_upper_bound": lower.greedy_upper_bound,
            "nodes": lower.nodes,
            "elapsed_seconds": lower.elapsed,
        },
        "partition_16": [
            {
                "disk": center_labels[i],
                "points": [labels[j] for j in group],
            }
            for i, group in enumerate(groups)
        ],
        "points": [
            {"index": i, "label": labels[i], "x": float(p[0]), "y": float(p[1])}
            for i, p in enumerate(points)
        ],
        "maximal_cliques": [
            [labels[i] for i in range(56) if (mask >> i) & 1]
            for mask in cliques
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("data/outer_partition_certificate.json"))
    args = parser.parse_args()
    result = verify(args.output)
    print(f"lambda = {result['lambda']:.15f}")
    print(f"points = {result['point_count']}")
    print(f"maximal cliques = {result['maximal_clique_count']} "
          f"{result['maximal_clique_sizes']}")
    print(f"15-group search = {result['search_15']}")
    print("explicit upper bound = 16 groups")
    print(f"certificate written to {args.output}")


if __name__ == "__main__":
    main()
