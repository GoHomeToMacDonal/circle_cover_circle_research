"""Bootstrapped rigorous upper bounds on r(n) for n = 11..25.

The radial programme of `radial_bound.py` needs, as input data, lower bounds
nmin(x) on the number of unit disks required to cover disk(0,x).  Previously we
fed it only

  * the PROVED values r(1..10)  (Bezdek; G. Fejes Toth), and
  * the certified fractional/measure bound  nmin(x) >= ceil(phi(x)).

But the radial programme itself *produces* bounds of exactly that kind for larger
n.  So we can bootstrap: compute an upper bound rho(n) on r(n) for n = 11, 12,
..., using the currently known nmin, then feed the improved
nmin(x) >= n+1 for x > rho(n) back in and repeat.  Each pass can only tighten the
constraints, so the sequence of bounds is monotone and converges.

Everything used is a relaxation of a necessary condition, so every bound produced
is a rigorous upper bound on r(n):

    (A) ARC     sum_i alpha(t, rho_i) >= pi          for all t in (0, R]
    (B) NESTING #{i : rho_i < d} >= nmin(d-1)        for all d
    (C) ORIGIN  min_i rho_i <= 1

with alpha replaced by its cell-wise maximum over a partition of [0, R+1] into
radial cells (a relaxation) and the constraints imposed on finite grids of t and
d (also a relaxation).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from arc_bound import alpha_max_on_cells

# Proved optimal covering radii, rounded DOWN so that "x > value" stays valid.
R_PROVEN = {
    1: 1.0, 2: 1.0, 3: 1.1547005, 4: 1.4142135, 5: 1.6411780,
    6: 1.7989070, 7: 2.0, 8: 2.2469790, 9: 2.4142135, 10: 2.5326880,
}


class Bounds_nmin:
    """nmin(x) >= value, assembled from proved values, phi, and bootstrapped rho."""

    def __init__(self, phi_table: dict[float, float]):
        self.rho: dict[int, float] = dict(R_PROVEN)
        self.phi_table = phi_table          # x -> certified fractional value

    def phi_at(self, x: float) -> float:
        xs = sorted(self.phi_table)
        if x <= xs[0]:
            return self.phi_table[xs[0]]
        if x >= xs[-1]:
            return self.phi_table[xs[-1]]
        # phi is increasing; interpolate conservatively (take the value at the
        # largest tabulated x that is <= our x)
        best = 0.0
        for t in xs:
            if t <= x:
                best = self.phi_table[t]
        return best

    def __call__(self, x: float) -> int:
        if x <= 0:
            return 0
        m = 1
        for n, v in sorted(self.rho.items()):
            if x > v + 1e-12:
                m = max(m, n + 1)
        return max(m, int(math.ceil(self.phi_at(x) - 1e-9)))


def radial_feasible(R: float, n: int, nmin, n_cells: int = 300, n_t: int = 400,
                    n_d: int = 90, time_limit: float = 60.0) -> int:
    """milp status: 0 feasible, 2 infeasible, 1 time limit."""
    edges = np.linspace(0.0, R + 1.0, n_cells + 1)
    lo = edges[:-1]
    ts = np.linspace(R / n_t, R, n_t)
    A_arc = np.array([alpha_max_on_cells(t, edges) for t in ts])

    cons = [
        LinearConstraint(A_arc, np.full(len(ts), math.pi), np.full(len(ts), np.inf)),
        LinearConstraint(np.ones((1, n_cells)), [n], [n]),
        LinearConstraint((lo <= 1.0).astype(float).reshape(1, -1), [1.0], [float(n)]),
    ]
    rows, ub = [], []
    for d in np.linspace(1.0, R + 1.0, n_d):
        cap = n - nmin(d - 1.0)
        if cap >= n:
            continue
        row = (lo >= d - 1e-15).astype(float)
        if row.sum() == 0:
            continue
        rows.append(row)
        ub.append(float(max(cap, 0)))
    if rows:
        cons.append(LinearConstraint(np.array(rows),
                                     np.full(len(ub), -np.inf), np.array(ub)))
    res = milp(c=np.zeros(n_cells), constraints=cons,
               integrality=np.ones(n_cells), bounds=Bounds(0, n),
               options={"time_limit": time_limit, "presolve": True})
    return res.status


def threshold(n: int, nmin, lo: float, hi: float, iters: int = 12, **kw) -> float:
    """Smallest R (to within the bisection) at which the radial programme is
    infeasible, i.e. a rigorous upper bound on r(n)."""
    # make sure hi is infeasible
    for _ in range(6):
        if radial_feasible(hi, n, nmin, **kw) == 2:
            break
        hi += 0.25
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        st = radial_feasible(mid, n, nmin, **kw)
        if st == 2:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> None:
    from measure_bound import phi

    print("Tabulating the certified fractional bound phi(x) ...")
    phi_table: dict[float, float] = {}
    for x in np.arange(1.0, 5.01, 0.1):
        r = phi(round(float(x), 3), n_t=320, n_sigma=1400, n_verify=40_000)
        phi_table[round(float(x), 3)] = r["certified"] if r else 0.0
    print("   x   phi(x)")
    for x in sorted(phi_table):
        if abs(x * 10 - round(x * 10)) < 1e-9 and int(round(x * 10)) % 5 == 0:
            print(f"  {x:4.1f} {phi_table[x]:8.4f}")

    nb = Bounds_nmin(phi_table)
    print("\nBootstrap passes (rigorous upper bounds rho(n) >= r(n)):")
    prev: dict[int, float] = {}
    for pass_no in range(3):
        changed = False
        line = []
        for n in range(11, 26):
            lo = nb.rho.get(n - 1, 1.0)
            hi = prev.get(n, lo + 1.2)
            th = threshold(n, nb, lo=lo, hi=hi + 0.05)
            old = nb.rho.get(n)
            if old is None or th < old - 1e-6:
                changed = True
            nb.rho[n] = min(th, old) if old else th
            prev[n] = nb.rho[n]
            line.append(f"r({n})<={nb.rho[n]:.4f}")
        print(f"  pass {pass_no}: " + "  ".join(line[:8]))
        print(f"           " + "  ".join(line[8:]))
        if not changed:
            print("  (converged)")
            break

    R25 = math.sqrt(6) + math.sqrt(3)
    print(f"\nFINAL: r(25) <= {nb.rho[25]:.6f}")
    print(f"       target sqrt6+sqrt3 = {R25:.6f}   gap = {nb.rho[25]-R25:.6f}")


if __name__ == "__main__":
    main()
