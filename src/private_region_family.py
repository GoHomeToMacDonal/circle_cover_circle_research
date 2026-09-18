"""Parametric finite point family from the 25 private-region boundaries.

The geometry is kept in the original scale: the known disks have radius one
and cover ``disk(0, R25)``.  For an owner centre ``c`` and an incident tight
point ``p``, define

    y(rho) = c + rho * (p - c),       0 < rho <= 1.

Thus every owner-tagged group lies on a circle of radius ``rho``.  Multiplying
by ``LAMBDA = 1 / R25`` gives the unit-target-disk normalization used in
``docs/finite_unique_cover_plan.md``.

At rho < 1 owner-tagged copies are intentionally retained: tight points shared
by several original circles move in different directions and become distinct.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

from config25 import LABELS, R25, centers25

LAMBDA = 1.0 / R25
SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class TaggedPoint:
    owner: str
    vertex: str
    point: tuple[float, float]


def _rotate(point: tuple[float, float], k: int) -> np.ndarray:
    angle = k * math.pi / 4.0
    c, s = math.cos(angle), math.sin(angle)
    x, y = point
    return np.array([c * x - s * y, s * x + c * y])


def tight_point_table() -> dict[str, np.ndarray]:
    """Return all 48 distinct tight points by their symbolic labels."""
    q = SQRT2
    bases = {
        "I": (1.0, 0.0),
        "M": (1.0 + q, 0.0),
        "T-": (2.0 + q / 2.0, q / 2.0),
        "T+": (1.0 + q, q),
        "E-": (2.0 + 3.0 * q / 2.0, q / 2.0),
        "E+": (2.0 + q, 1.0 + q),
    }
    return {f"{kind}{k}": _rotate(base, k)
            for kind, base in bases.items() for k in range(8)}


def owner_vertices() -> dict[str, tuple[str, ...]]:
    """Cyclic private-boundary vertex incidences for all 25 owner disks."""
    result: dict[str, tuple[str, ...]] = {
        "O": tuple(f"I{k}" for k in range(8)),
    }
    for k in range(8):
        km = (k - 1) % 8
        kp = (k + 1) % 8
        result[f"A{k}"] = (
            f"I{k}", f"I{kp}", f"M{kp}", f"T+{k}", f"T-{k}", f"M{k}",
        )
        result[f"B{k}"] = (
            f"M{k}", f"T-{k}", f"E-{k}", f"E+{km}", f"T+{km}",
        )
        result[f"C{k}"] = (f"T-{k}", f"T+{k}", f"E+{k}", f"E-{k}")
    return result


def family(rho: float, *, normalized: bool = False) -> tuple[np.ndarray, list[str], list[str], list[list[int]]]:
    """Generate Q_rho, owner labels, vertex labels, and the 25 owner groups."""
    if not (0.0 < rho <= 1.0):
        raise ValueError("rho must satisfy 0 < rho <= 1")
    centers = centers25()
    center_by_label = dict(zip(LABELS, centers))
    vertices = tight_point_table()
    incidence = owner_vertices()
    points: list[np.ndarray] = []
    owners: list[str] = []
    names: list[str] = []
    groups: list[list[int]] = []
    for owner in LABELS:
        c = center_by_label[owner]
        group: list[int] = []
        for vertex in incidence[owner]:
            p = c + rho * (vertices[vertex] - c)
            group.append(len(points))
            points.append(p * (LAMBDA if normalized else 1.0))
            owners.append(owner)
            names.append(vertex)
        groups.append(group)
    return np.asarray(points), owners, names, groups



def boundary_dense_family(
    rho: float,
    level: int,
    *,
    normalized: bool = False,
) -> tuple[np.ndarray, list[str], list[str], list[list[int]]]:
    """Add nested dyadic samples on every owner circle inside the target disk.

    ``level=0`` is exactly :func:`family`.  At positive levels, circles wholly
    inside ``disk(0,R25)`` are subdivided between cyclic original directions.
    For a partially clipped B/C circle, the two intersections with the target
    boundary are added and only the single inward feasible arc is subdivided.
    Level ``ell`` cuts every anchored arc into ``2**ell`` pieces, so the point
    sets are nested as the level increases.
    """
    if level < 0:
        raise ValueError("level must be nonnegative")
    base_points, owners, names, groups = family(rho, normalized=False)
    if level == 0:
        if normalized:
            return base_points * LAMBDA, owners, names, groups
        return base_points, owners, names, groups

    points = [point.copy() for point in base_points]
    dense_owners = list(owners)
    dense_names = list(names)
    dense_groups = [list(group) for group in groups]
    centers = centers25()
    pieces = 1 << level
    two_pi = 2.0 * math.pi

    def add(owner_index: int, angle: float, name: str) -> None:
        center = centers[owner_index]
        point = center + rho * np.array([math.cos(angle), math.sin(angle)])
        # Roundoff at a target-boundary endpoint can be a few ulps outward.
        radius = float(np.linalg.norm(point))
        if radius > R25 and radius - R25 < 2e-13:
            point *= R25 / radius
        if np.linalg.norm(point - center) > rho + 3e-13:
            raise AssertionError("dense point left its owner circle")
        if np.linalg.norm(point) > R25 + 3e-13:
            raise AssertionError("dense point left the target disk")
        dense_groups[owner_index].append(len(points))
        points.append(point)
        dense_owners.append(LABELS[owner_index])
        dense_names.append(name)

    for owner_index, (center, group) in enumerate(zip(centers, groups)):
        original_angles = [
            math.atan2(*(base_points[index] - center)[::-1]) % two_pi
            for index in group
        ]
        center_radius = float(np.linalg.norm(center))
        segments: list[tuple[float, float, str]] = []
        if center_radius + rho <= R25 + 2e-13:
            anchors = sorted(original_angles)
            for segment, start in enumerate(anchors):
                end = anchors[(segment + 1) % len(anchors)]
                if end <= start:
                    end += two_pi
                segments.append((start, end, f"arc{segment}"))
        else:
            radial = math.atan2(center[1], center[0])
            cosine = (R25 * R25 - center_radius * center_radius - rho * rho) / (
                2.0 * center_radius * rho
            )
            alpha = math.acos(max(-1.0, min(1.0, cosine)))
            relative = sorted((angle - radial) % two_pi for angle in original_angles)
            if relative[0] < alpha - 2e-12 or relative[-1] > two_pi - alpha + 2e-12:
                raise AssertionError("original anchor lies outside feasible owner arc")
            anchors = [alpha]
            for angle in relative:
                if abs(angle - anchors[-1]) > 2e-12:
                    anchors.append(angle)
            if abs((two_pi - alpha) - anchors[-1]) > 2e-12:
                anchors.append(two_pi - alpha)
            # Add the current-rho target-boundary intersections when they are
            # not already original anchors (at rho=1 they coincide with E).
            for endpoint, suffix in ((anchors[0], "target_start"),
                                     (anchors[-1], "target_end")):
                if all(abs(endpoint - angle) > 2e-12 for angle in relative):
                    add(owner_index, radial + endpoint, suffix)
            for segment, (start, end) in enumerate(zip(anchors, anchors[1:])):
                segments.append((radial + start, radial + end, f"arc{segment}"))

        for start, end, segment_name in segments:
            for numerator in range(1, pieces):
                angle = start + (end - start) * numerator / pieces
                add(owner_index, angle,
                    f"{segment_name}:{numerator}/{pieces}")

    result = np.asarray(points)
    if normalized:
        result = result * LAMBDA
    return result, dense_owners, dense_names, dense_groups



def boundary_with_core_family(
    rho: float,
    level: int,
    core_scale: float,
    *,
    normalized: bool = False,
) -> tuple[np.ndarray, list[str], list[str], list[list[int]]]:
    """Add a small homothetic copy of each owner's outer boundary shape.

    For every owner-tagged outer point ``p = c + rho*u``, add
    ``c + core_scale*(p-c)``.  The copy has the same directions and cyclic
    shape, with circumradius ``core_scale*rho``.  ``core_scale=0`` is defined
    as the no-core baseline rather than 25 collapsed centre points.
    """
    if not (0.0 <= core_scale < 1.0):
        raise ValueError("core_scale must satisfy 0 <= core_scale < 1")
    outer, owners, names, groups = boundary_dense_family(
        rho, level, normalized=False
    )
    if core_scale == 0.0:
        if normalized:
            return outer * LAMBDA, owners, names, groups
        return outer, owners, names, groups

    points = [point.copy() for point in outer]
    result_owners = list(owners)
    result_names = list(names)
    result_groups = [list(group) for group in groups]
    center_by_owner = dict(zip(LABELS, centers25()))
    for index, (point, owner, name) in enumerate(zip(outer, owners, names)):
        center = center_by_owner[owner]
        core = center + core_scale * (point - center)
        if np.linalg.norm(core) > R25 + 3e-13:
            raise AssertionError("core point left the target disk")
        owner_index = LABELS.index(owner)
        result_groups[owner_index].append(len(points))
        points.append(core)
        result_owners.append(owner)
        result_names.append(f"core:{name}")

    result = np.asarray(points)
    if normalized:
        result *= LAMBDA
    return result, result_owners, result_names, result_groups


def boundary_with_radial_layers_family(
    rho: float,
    level: int,
    radial_scales: tuple[float, ...] | list[float],
    *,
    normalized: bool = False,
) -> tuple[np.ndarray, list[str], list[str], list[list[int]]]:
    """Add concentric homothetic layers whose aligned points form spokes.

    The outer scale 1 is always supplied by :func:`boundary_dense_family` and
    must not be listed.  Every positive scale ``s < 1`` adds ``c+s*(p-c)`` for
    each owner-tagged outer point.  Scale zero degenerates to one centre marker
    per owner rather than repeated copies for all directions.
    """
    scales = tuple(sorted(float(scale) for scale in radial_scales))
    if len(scales) != len(set(scales)):
        raise ValueError("radial scales must be distinct")
    if any(not 0.0 <= scale < 1.0 for scale in scales):
        raise ValueError("radial scales must satisfy 0 <= scale < 1")

    outer, owners, names, groups = boundary_dense_family(
        rho, level, normalized=False
    )
    if not scales:
        if normalized:
            return outer * LAMBDA, owners, names, groups
        return outer, owners, names, groups

    points = [point.copy() for point in outer]
    result_owners = list(owners)
    result_names = list(names)
    result_groups = [list(group) for group in groups]
    centers = centers25()
    owner_index = {owner: index for index, owner in enumerate(LABELS)}

    for scale in scales:
        if scale == 0.0:
            for index, (owner, center) in enumerate(zip(LABELS, centers)):
                result_groups[index].append(len(points))
                points.append(center.copy())
                result_owners.append(owner)
                result_names.append("layer:0:center")
            continue
        for point, owner, name in zip(outer, owners, names):
            index = owner_index[owner]
            center = centers[index]
            layered = center + scale * (point - center)
            if np.linalg.norm(layered) > R25 + 3e-13:
                raise AssertionError("radial-layer point left the target disk")
            result_groups[index].append(len(points))
            points.append(layered)
            result_owners.append(owner)
            result_names.append(f"layer:{scale:.12g}:{name}")

    result = np.asarray(points)
    if normalized:
        result *= LAMBDA
    return result, result_owners, result_names, result_groups
def diagnostics(rho: float) -> dict[str, float | int]:
    points, owners, _, groups = family(rho)
    centers = centers25()
    max_circle_error = 0.0
    min_hull_margin = math.inf
    for center, group in zip(centers, groups):
        vectors = points[group] - center
        max_circle_error = max(max_circle_error,
                               float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - rho))))
        angles = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]) % (2.0 * math.pi))
        gaps = np.diff(np.r_[angles, angles[0] + 2.0 * math.pi])
        # Positive iff the owner centre is strictly inside the convex hull.
        min_hull_margin = min(min_hull_margin, math.pi - float(np.max(gaps)))
    normalized_points = points * LAMBDA
    return {
        "rho": rho,
        "points_with_owner_multiplicity": len(points),
        "distinct_rounded_points": len({(round(float(x), 12), round(float(y), 12))
                                        for x, y in points}),
        "groups": len(groups),
        "max_owner_circle_error": max_circle_error,
        "min_convex_hull_angular_margin": min_hull_margin,
        "max_normalized_point_radius": float(np.linalg.norm(normalized_points, axis=1).max()),
        "unit_disk_margin": 1.0 - float(np.linalg.norm(normalized_points, axis=1).max()),
        "owner_count": len(set(owners)),
    }


def boundary_arcs() -> dict[str, tuple[str, ...]]:
    """Carrier circle for each arc after the correspondingly ordered vertex.

    The final entry joins the final vertex back to the first.  ``S_R`` denotes
    the boundary of the target disk; all other labels name neighbouring cover
    circles.  Together with :func:`owner_vertices`, this is the complete D8
    private-cell boundary combinatorics.
    """
    result: dict[str, tuple[str, ...]] = {
        "O": tuple(f"A{k}" for k in range(8)),
    }
    for k in range(8):
        km = (k - 1) % 8
        kp = (k + 1) % 8
        result[f"A{k}"] = ("O", f"A{kp}", f"B{kp}", f"C{k}", f"B{k}", f"A{km}")
        result[f"B{k}"] = (f"A{k}", f"C{k}", "S_R", f"C{km}", f"A{km}")
        result[f"C{k}"] = (f"A{k}", f"B{kp}", "S_R", f"B{k}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.99])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    reports = [diagnostics(rho) for rho in args.rho]
    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
        return
    for report in reports:
        print(" ".join(f"{key}={value}" for key, value in report.items()), flush=True)


if __name__ == "__main__":
    main()
