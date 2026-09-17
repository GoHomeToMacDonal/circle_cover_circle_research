"""Theorem 2: strict local optimality of the n=25 configuration in R^50.

Setup.  C* = (c_1*,...,c_25*) is the configuration of config25, R* = sqrt6+sqrt3,
and T = {p_1,...,p_48} is the set of *tight points*: |p_j| <= R* and
min_i |p_j - c_i*| = 1.  For each j let

    O_j = { i : |p_j - c_i*| = 1 }        (the owners of p_j)
    n_ij = p_j - c_i*        for i in O_j (a unit vector, the outward normal).

Key elementary fact.  If c_i = c_i* + w_i and w_i . n_ij <= 0 then

    |p_j - c_i|^2 = |p_j - c_i*|^2 - 2 w_i . n_ij + |w_i|^2 >= 1,

so p_j is *not* in the open unit disk about c_i, and a fortiori not in the
closed disk of any radius rho < 1 about c_i.

Consequence.  Suppose some configuration C covers disk(0, R') with R' = lambda R*,
lambda > 1.  Rescaling by 1/lambda, the 25 disks of radius rho = 1/lambda < 1
centred at c_i/lambda cover disk(0, R*), hence cover every tight point.  Writing
w_i = c_i/lambda - c_i*, the fact above forces, for every j,

    (*)   there is i in O_j with w_i . n_ij > 0.

So if NO vector w = (w_1,...,w_25) in R^50 satisfies (*) for all j, no such C can
exist, i.e. R* is a strict local maximum -- and the "local" is only used to keep
the *non*-owners away, which costs an explicit epsilon_0 (Step 3).

Deciding (*) is a finite combinatorial problem.  By Gordan's theorem a finite set
N of vectors admits w with w.n > 0 for all n in N iff 0 is not in conv(N).  So (*)
is solvable iff there is an assignment j -> i(j) in O_j such that for every disk i
the assigned normal set N_i = {n_ij : i(j) = i} has 0 not in conv(N_i).  In R^2
the minimal sets containing 0 in their convex hull have 2 elements (antipodal) or
3 elements (positively spanning), so the condition is a small set of "forbidden
subset" constraints and the whole question is a tiny 0/1 feasibility program.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint

from config25 import LABELS, R25, centers25, dedup, tight_points

TOL = 1e-9


# --------------------------------------------------------------------------- #
def build_incidence(tol: float = 1e-7):
    """Tight points, their owners, and the outward normals."""
    C = centers25()
    P, _, _ = tight_points(C, R25, tol=1e-9)
    P = dedup(P, tol=1e-7)
    owners: list[list[int]] = []
    normals: list[dict[int, np.ndarray]] = []
    for p in P:
        d = np.linalg.norm(p - C, axis=1)
        own = [int(i) for i in np.nonzero(np.abs(d - 1.0) < tol)[0]]
        owners.append(own)
        normals.append({i: (p - C[i]) / np.linalg.norm(p - C[i]) for i in own})
    return C, P, owners, normals


def in_open_halfplane(vecs: np.ndarray) -> bool:
    """True iff some w has w.v > 0 for every row v (equivalently 0 notin conv)."""
    if len(vecs) == 0:
        return True
    m = len(vecs)
    # LP: maximise s subject to  v_k . w >= s,  |w|_inf <= 1
    #   feasible with s > 0  <=>  open halfplane
    A = np.hstack([-vecs, np.ones((m, 1))])          # -v.w + s <= 0
    res = linprog(
        c=[0, 0, -1],
        A_ub=A,
        b_ub=np.zeros(m),
        bounds=[(-1, 1), (-1, 1), (0, 1)],
        method="highs",
    )
    return bool(res.success and -res.fun > 1e-9)


def minimal_forbidden(vecs: np.ndarray) -> list[tuple[int, ...]]:
    """Minimal index subsets whose vectors do NOT fit in an open halfplane."""
    n = len(vecs)
    bad: list[tuple[int, ...]] = []
    for size in (2, 3):
        for comb in itertools.combinations(range(n), size):
            sub = vecs[list(comb)]
            if in_open_halfplane(sub):
                continue
            if any(set(b) <= set(comb) for b in bad):
                continue
            bad.append(comb)
    return bad


# --------------------------------------------------------------------------- #
def solve_assignment(owners, normals, n_disks: int = 25, verbose: bool = True):
    """Decide feasibility of (*): is there an assignment with all cones open?

    Returns (feasible, info).
    """
    # variables x[(j,i)] for i in owners[j]
    var = [(j, i) for j, own in enumerate(owners) for i in own]
    idx = {k: t for t, k in enumerate(var)}
    nv = len(var)

    A_rows, lo, hi = [], [], []
    # each tight point served at least once
    for j, own in enumerate(owners):
        row = np.zeros(nv)
        for i in own:
            row[idx[(j, i)]] = 1.0
        A_rows.append(row)
        lo.append(1.0)
        hi.append(np.inf)

    # per-disk cone constraints
    n_cone = 0
    for i in range(n_disks):
        js = [j for j, own in enumerate(owners) if i in own]
        if len(js) < 2:
            continue
        vecs = np.array([normals[j][i] for j in js])
        for comb in minimal_forbidden(vecs):
            row = np.zeros(nv)
            for k in comb:
                row[idx[(js[k], i)]] = 1.0
            A_rows.append(row)
            lo.append(-np.inf)
            hi.append(float(len(comb) - 1))
            n_cone += 1
    if verbose:
        print(f"  variables {nv}, point constraints {len(owners)}, "
              f"cone constraints {n_cone}")

    A = np.array(A_rows)
    res = milp(
        c=np.zeros(nv),
        constraints=LinearConstraint(A, lo, hi),
        integrality=np.ones(nv),
        bounds=Bounds(0, 1),
    )
    return bool(res.success), res


# --------------------------------------------------------------------------- #
def epsilon0(C: np.ndarray, P: np.ndarray, owners) -> float:
    """How far a non-owner is from a tight point (gives the neighbourhood size)."""
    best = np.inf
    for j, p in enumerate(P):
        d = np.linalg.norm(p - C, axis=1)
        for i in range(len(C)):
            if i not in owners[j]:
                best = min(best, d[i] - 1.0)
    return float(best)


def main() -> None:
    print("=" * 76)
    print("Theorem 2.  Strict local optimality of the n=25 configuration")
    print("=" * 76)
    C, P, owners, normals = build_incidence()
    print(f"\ntight points: {len(P)}")
    from collections import Counter

    print("owner-set sizes:", dict(Counter(len(o) for o in owners)))
    cnt = Counter(i for o in owners for i in o)
    print("tight points owned per disk:")
    for i in range(25):
        print(f"    {LABELS[i]:>3}: {cnt[i]}")

    print("\nPer-disk normal cones (angles in degrees) and their spread:")
    for i in range(25):
        js = [j for j, o in enumerate(owners) if i in o]
        if not js:
            continue
        ang = sorted(math.degrees(math.atan2(*normals[j][i][::-1])) % 360 for j in js)
        vecs = np.array([normals[j][i] for j in js])
        ok = in_open_halfplane(vecs)
        print(f"    {LABELS[i]:>3}: {[f'{a:7.3f}' for a in ang]}  "
              f"all-in-open-halfplane={ok}")

    print("\nDeciding the assignment program (*)")
    feasible, res = solve_assignment(owners, normals)
    print(f"\n  feasible = {feasible}")
    if feasible:
        print("  status:", res.message)
        var = [(j, i) for j, own in enumerate(owners) for i in own]
        chosen = {}
        for t, (j, i) in enumerate(var):
            if res.x[t] > 0.5:
                chosen.setdefault(i, []).append(j)
        print("  a feasible assignment (disk -> served tight points):")
        for i, js in sorted(chosen.items()):
            print(f"    {LABELS[i]:>3}: {js}")
        print("\n  => the first-order tight-point test does NOT by itself rule out")
        print("     improving perturbations; a finer certificate set is needed.")
    else:
        e0 = epsilon0(C, P, owners)
        print(f"  => NO direction w can keep all 48 tight points covered.")
        print(f"  non-owner slack epsilon_0 = {e0:.9f}")
        print("  THEOREM 2: R* is a strict local maximum of the covering radius.")


if __name__ == "__main__":
    main()
