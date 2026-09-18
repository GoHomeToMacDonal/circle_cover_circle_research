"""Exact countercertificate for the natural private-vertex Q_rho family.

Eighteen fixed centres in Q(sqrt(2)) cover all 48 geometric limit points Q_1
with radius

    h = cos(pi/8) = sqrt(2 + sqrt(2)) / 2 < 1.

Every owner-tagged point satisfies

    y(rho) = p + (1-rho)(c-p),  |c-p| = 1.

Consequently its distance to the same fixed centre is at most h + 1-rho.
For rho > rho0=(1+h)/2 this is strictly less than rho.  Thus this explicit
family has tau_25(Q_rho) < rho (or < rho*lambda after normalization) throughout
the whole tail interval; it cannot satisfy the plan's core proposition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from private_region_family import owner_vertices

Q = sp.sqrt(2)
H2 = (2 + Q) / 4
H = sp.sqrt(2 + Q) / 2
RHO0 = (1 + H) / 2


def rotate(point: tuple[sp.Expr, sp.Expr], k: int, quarter: bool = False) -> tuple[sp.Expr, sp.Expr]:
    angle = k * (sp.pi / 2 if quarter else sp.pi / 4)
    c, s = sp.expand_trig(sp.cos(angle)), sp.expand_trig(sp.sin(angle))
    x, y = point
    return sp.simplify(c * x - s * y), sp.simplify(s * x + c * y)


def exact_tight_points() -> dict[str, tuple[sp.Expr, sp.Expr]]:
    bases = {
        "I": (sp.Integer(1), sp.Integer(0)),
        "M": (1 + Q, sp.Integer(0)),
        "T-": (2 + Q / 2, Q / 2),
        "T+": (1 + Q, Q),
        "E-": (2 + 3 * Q / 2, Q / 2),
        "E+": (2 + Q, 1 + Q),
    }
    return {f"{name}{k}": rotate(point, k)
            for name, point in bases.items() for k in range(8)}


def exact_owner_centers() -> dict[str, tuple[sp.Expr, sp.Expr]]:
    result = {"O": (sp.Integer(0), sp.Integer(0))}
    for name, base in (
        ("A", (1 + Q / 2, Q / 2)),
        ("B", (2 + Q, sp.Integer(0))),
        ("C", (2 + Q, Q)),
    ):
        result.update({f"{name}{k}": rotate(base, k) for k in range(8)})
    return result


def counterexample_centers() -> list[tuple[sp.Expr, sp.Expr]]:
    first = (-sp.Rational(1, 2) + Q / 4, Q / 4)
    result = [first, (-first[0], -first[1])]
    bases = [
        (1 + 3 * Q / 4, sp.Rational(1, 2) + Q / 4),
        (3 * Q / 4, sp.Rational(3, 2) + 5 * Q / 4),
        (2 + 5 * Q / 4, sp.Rational(1, 2) + 3 * Q / 4),
        (1 + 3 * Q / 4, sp.Rational(3, 2) + 3 * Q / 4),
    ]
    for base in bases:
        result.extend(rotate(base, k, quarter=True) for k in range(4))
    return result


def squared_distance(a: tuple[sp.Expr, sp.Expr], b: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    return sp.simplify((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def verify() -> dict[str, object]:
    points = exact_tight_points()
    owners = exact_owner_centers()
    incidence = owner_vertices()
    centers = counterexample_centers()
    if len(points) != 48 or len(centers) != 18:
        raise AssertionError("unexpected certificate size")

    incidence_checks = 0
    for owner, names in incidence.items():
        for name in names:
            if sp.simplify(squared_distance(owners[owner], points[name]) - 1) != 0:
                raise AssertionError(f"non-unit owner incidence: {owner}:{name}")
            incidence_checks += 1
    if incidence_checks != 128:
        raise AssertionError("unexpected owner incidence count")

    assignments: dict[str, int] = {}
    boundary_points = 0
    strict_points = 0
    for name, point in points.items():
        differences = [sp.simplify(squared_distance(point, center) - H2)
                       for center in centers]
        covering = [i for i, value in enumerate(differences)
                    if value == 0 or value.is_negative]
        if not covering:
            raise AssertionError(f"uncovered exact tight point: {name}")
        assignments[name] = covering[0]
        if any(value == 0 for value in differences):
            boundary_points += 1
        else:
            strict_points += 1

    if sp.simplify(1 - H2) != (2 - Q) / 4 or not (H < 1):
        raise AssertionError("failed to prove h < 1")
    # Exact symbolic identity behind strictness on the open tail interval.
    rho = sp.Symbol("rho", real=True)
    strict_gap = sp.simplify(rho - (H + 1 - rho))
    if sp.simplify(strict_gap - 2 * (rho - RHO0)) != 0:
        raise AssertionError("tail-interval identity failed")

    return {
        "status": "exact_countercertificate_verified",
        "arithmetic": "sympy exact algebraic",
        "limit_geometric_points": len(points),
        "owner_tagged_incidences": incidence_checks,
        "fixed_cover_centers": len(centers),
        "covered_limit_points": len(assignments),
        "boundary_limit_points": boundary_points,
        "strict_limit_points": strict_points,
        "h": str(H),
        "h_decimal": float(sp.N(H, 18)),
        "rho0": str(RHO0),
        "rho0_decimal": float(sp.N(RHO0, 18)),
        "tail_radius": "h + 1 - rho",
        "strict_gap": str(strict_gap),
        "conclusion_original_scale": "tau_25(Q_rho) <= h+1-rho < rho for rho in (rho0,1)",
        "conclusion_normalized": "tau_25(lambda*Q_rho) < rho*lambda for rho in (rho0,1)",
        "centers": [[str(x), str(y)] for x, y in centers],
        "assignments": assignments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
