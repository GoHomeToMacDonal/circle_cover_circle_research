"""A rigorous upper bound on r(25) from radial data only.

Let 25 unit disks cover D_R = disk(0,R) and set rho_i = |c_i|, sorted.  Three
families of necessary conditions use nothing but the multiset {rho_i}:

(A) ARC CONDITION.  For every t in (0,R] the circle S_t must be covered, and
    B(c_i,1) cuts an arc of half-width alpha(t,rho_i), so

        sum_i alpha(t, rho_i) >= pi,
        alpha(t,rho) = arccos((t^2+rho^2-1)/(2 t rho)),  pi if t+rho<=1, 0 if |t-rho|>1.

(B) NESTING CONDITION.  A disk with rho_i >= d covers no point of the open disk
    of radius d-1.  Hence the disks with rho_i < d already cover disk(0,d-1), so

        #{i : rho_i < d}  >=  nmin(d-1)

    where nmin(x) is the minimum number of unit disks needed to cover disk(0,x).
    Equivalently #{i : rho_i >= d} <= 25 - nmin(d-1).  For nmin we use
      * the PROVEN optimal values r(n) for n <= 10 (Bezdek; G. Fejes Toth), giving
        nmin(x) >= n+1 whenever x > r(n);
      * our own certified fractional (measure) bound nmin(x) >= ceil(phi(x)).

(C) ORIGIN.  The origin is covered, so min_i rho_i <= 1.

RIGOROUS DISCRETISATION.  Partition [0,R+1] into cells and replace alpha by its
cell-wise maximum abar (attained at rho = sqrt(t^2-1) when that lies in the cell,
else at an endpoint).  Count variables n_r = #{i : rho_i in cell r}.  For (B) only
cells lying entirely above d are counted, which under-counts and so weakens the
constraint.  Both steps are relaxations, hence

        ILP infeasible  ==>  25 unit disks cannot cover D_R  ==>  r(25) < R.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from arc_bound import alpha, alpha_max_on_cells
from measure_bound import phi

R25 = math.sqrt(6) + math.sqrt(3)

# Proven optimal covering radii r(n) for n <= 10 (Friedman's table; the entries
# marked "proved" there).  r(2) = r(1) = 1.
R_PROVEN = {
    1: 1.0,
    2: 1.0,
    3: 2 / math.sqrt(3),
    4: math.sqrt(2),
    5: 1.641_178,          # Bezdek 1983  (rounded DOWN: a valid lower bound on r(5))
    6: 1.798_907,          # Bezdek 1979  (rounded DOWN)
    7: 2.0,
    8: 2.246_979,          # G. Fejes Toth 1996 (rounded DOWN)
    9: 1 + math.sqrt(2),
    10: 2.532_688,         # G. Fejes Toth 2005 (rounded DOWN)
}


def nmin_proven(x: float) -> int:
    """Largest m with the guarantee: covering disk(0,x) needs >= m unit disks."""
    m = 1
    for n in sorted(R_PROVEN):
        if x > R_PROVEN[n] + 1e-12:
            m = n + 1
    return m


_PHI_CACHE: dict[float, float] = {}


def nmin_measure(x: float) -> int:
    """ceil of the certified fractional covering number of disk(0,x)."""
    key = round(x, 4)
    if key not in _PHI_CACHE:
        r = phi(key, n_t=350, n_sigma=1500, n_verify=40_000)
        _PHI_CACHE[key] = r["certified"] if r else 0.0
    v = _PHI_CACHE[key]
    return int(math.ceil(v - 1e-9))


def nmin(x: float) -> int:
    if x <= 0:
        return 0
    return max(nmin_proven(x), nmin_measure(x))


# --------------------------------------------------------------------------- #
def build_program(R: float, n_cells: int, n_t: int, n_d: int):
    edges = np.linspace(0.0, R + 1.0, n_cells + 1)
    lo = edges[:-1]

    ts = np.linspace(R / n_t, R, n_t)
    A_arc = np.array([alpha_max_on_cells(t, edges) for t in ts])

    rows, ub, info = [], [], []
    for d in np.linspace(1.0, R + 1.0, n_d):
        need = nmin(d - 1.0)
        cap = 25 - need
        if cap >= 25:
            continue
        row = (lo >= d - 1e-15).astype(float)
        if row.sum() == 0:
            continue
        rows.append(row)
        ub.append(float(cap))
        info.append((d, need, cap))
    A_nest = np.array(rows) if rows else np.zeros((0, n_cells))
    return edges, ts, A_arc, A_nest, np.array(ub), info


def feasible(R: float, n_cells: int = 400, n_t: int = 600, n_d: int = 120,
             time_limit: float = 120.0, verbose: bool = False):
    edges, ts, A_arc, A_nest, ub, info = build_program(R, n_cells, n_t, n_d)
    K = n_cells
    cons = [
        LinearConstraint(A_arc, np.full(len(ts), math.pi), np.full(len(ts), np.inf)),
        LinearConstraint(np.ones((1, K)), [25], [25]),
    ]
    if len(A_nest):
        cons.append(LinearConstraint(A_nest, np.full(len(ub), -np.inf), ub))
    # (C) origin covered: at least one rho_i <= 1
    row = (edges[:-1] <= 1.0).astype(float)
    cons.append(LinearConstraint(row.reshape(1, -1), [1.0], [25.0]))
    res = milp(
        c=np.zeros(K),
        constraints=cons,
        integrality=np.ones(K),
        bounds=Bounds(0, 25),
        options={"time_limit": time_limit, "presolve": True},
    )
    if verbose and res.status == 0:
        n = np.round(res.x).astype(int)
        nz = np.nonzero(n)[0]
        print("      feasible radial profile:",
              [(round(float(edges[i]), 4), int(n[i])) for i in nz])
    return res


def check_true_config(R: float = R25) -> None:
    print("Sanity check: does the true n=25 configuration satisfy (A),(B),(C)?")
    rho = np.array([0.0] + [2 * math.cos(math.pi / 8)] * 8
                   + [2 + math.sqrt(2)] * 8 + [4 * math.cos(math.pi / 8)] * 8)
    worst = min(float(alpha(t, rho).sum()) - math.pi
                for t in np.linspace(1e-4, R, 20001))
    print(f"  (A) min_t (sum alpha - pi) = {worst:+.3e}   (must be >= 0)")
    print("  (B) nesting:")
    for d in (2.0, 2.4143, 2.8, 3.0, 3.2466, 3.41422, 3.5, 3.6956, 3.9):
        cnt = int((rho >= d - 1e-12).sum())
        need = nmin(d - 1.0)
        flag = "OK" if cnt <= 25 - need else "VIOLATED"
        print(f"      d={d:7.4f}: #\u007brho>=d\u007d={cnt:2d}  nmin({d-1:.4f})={need:2d}"
              f"  cap={25-need:2d}  {flag}")
    print(f"  (C) min rho = {rho.min():.4f} <= 1  OK")


def main() -> None:
    check_true_config()
    print("\nnmin table (proven small-n values vs certified measure bound):")
    for x in [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.42, 2.53, 2.7, 3.0, 3.2, 3.5]:
        print(f"  x={x:5.2f}  proven>={nmin_proven(x):2d}  measure>={nmin_measure(x):2d}"
              f"  => nmin>={nmin(x):2d}")

    print(f"\n{'='*78}\nfeasibility of the radial program\n{'='*78}")
    for R in (4.19, 4.22, 4.25, 4.28, 4.30, 4.3622):
        res = feasible(R, verbose=True)
        tag = {0: "FEASIBLE", 2: "INFEASIBLE -> r(25) < R", 1: "time limit"}.get(
            res.status, str(res.status))
        print(f"  R = {R:.6f}:  {tag}")


if __name__ == "__main__":
    main()
