"""The radial-arc necessary condition, and its exact integer strengthening.

NECESSARY CONDITION.  Let 25 unit disks with centres c_1..c_25 cover disk(0,R),
and put rho_i = |c_i|.  For every t in (0,R] the circle S_t must be covered, and
the disk B(c_i,1) meets S_t in a closed arc of angular half-width

    alpha(t, rho) = arccos((t^2 + rho^2 - 1)/(2 t rho))  in [0, pi],
    alpha = pi if t + rho <= 1,   alpha = 0 if |t - rho| > 1.

Total arc length must be at least the full circle, so

    (ARC)      sum_i alpha(t, rho_i) >= pi        for every t in (0, R].

(ARC) involves the 25 radial distances only -- all angular information is thrown
away -- yet it is already quite restrictive.  Note

    max_rho alpha(t, rho) = arcsin(1/t)   attained at rho = sqrt(t^2 - 1),

so (ARC) at t = R alone forces at least ceil(pi / arcsin(1/R)) disks.

RELATION TO THE MEASURE BOUND.  Averaging (ARC) against a measure nu on (0,R]
gives exactly the rotation-invariant measure ("probabilistic") bound, so that
bound is precisely the LP relaxation of (ARC), obtained by letting the 25 radii be
a continuous mass distribution.  Keeping the count integral is strictly stronger.

RIGOROUS INTEGER RELAXATION.  Partition [0, R+1] into cells J_1..J_K and let

    abar(t, J) = max_{rho in J} alpha(t, rho)

(computable in closed form: alpha(t, .) increases up to rho = sqrt(t^2-1) and
decreases after, so the max is at that point if it lies in J and at an endpoint
otherwise).  If rho_i lies in cell J_{r(i)} and n_r = #{i : r(i) = r}, then (ARC)
implies

    (ILP)   sum_r n_r abar(t_j, r) >= pi   for every t_j in a finite grid,
            sum_r n_r = 25,  n_r in Z_{>=0}.

Both discretisations are *relaxations* (cells enlarge alpha; fewer t's means fewer
constraints), so

    (ILP) infeasible  ==>  (ARC) infeasible  ==>  25 unit disks cannot cover
                                                   disk(0,R)  ==>  r(25) < R.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

R25 = math.sqrt(6) + math.sqrt(3)


# --------------------------------------------------------------------------- #
def alpha(t: float, rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    out = np.zeros_like(rho)
    if t <= 0:
        return out
    swallow = t + rho <= 1.0
    out[swallow] = math.pi
    live = (~swallow) & (np.abs(t - rho) <= 1.0) & (rho > 1e-15)
    c = np.clip((t * t + rho[live] ** 2 - 1.0) / (2 * t * rho[live]), -1.0, 1.0)
    out[live] = np.arccos(c)
    if t <= 1.0:
        out[rho <= 1e-15] = math.pi
    return out


def alpha_max_on_cells(t: float, edges: np.ndarray) -> np.ndarray:
    """max of alpha(t, .) over each cell [edges[k], edges[k+1]]."""
    lo, hi = edges[:-1], edges[1:]
    vals = np.maximum(alpha(t, lo), alpha(t, hi))
    if t > 1.0:
        star = math.sqrt(t * t - 1.0)
        inside = (lo <= star) & (star <= hi)
        vals[inside] = math.asin(1.0 / t)
    else:
        # alpha = pi is attained wherever rho <= 1 - t
        vals[lo <= 1.0 - t] = math.pi
    return vals


def build(R: float, n_cells: int, n_t: int):
    edges = np.linspace(0.0, R + 1.0, n_cells + 1)
    ts = np.linspace(R / n_t, R, n_t)
    A = np.array([alpha_max_on_cells(t, edges) for t in ts])
    return edges, ts, A


# --------------------------------------------------------------------------- #
def lp_feasible(A: np.ndarray, n: int = 25) -> bool:
    """LP relaxation of (ILP): continuous radii distribution."""
    K = A.shape[1]
    res = linprog(
        c=np.zeros(K),
        A_ub=-A,
        b_ub=-np.full(A.shape[0], math.pi),
        A_eq=np.ones((1, K)),
        b_eq=[n],
        bounds=[(0, None)] * K,
        method="highs",
    )
    return bool(res.success)


def ilp_feasible(A: np.ndarray, n: int = 25, time_limit: float = 600.0):
    K = A.shape[1]
    cons = [
        LinearConstraint(A, np.full(A.shape[0], math.pi), np.full(A.shape[0], np.inf)),
        LinearConstraint(np.ones((1, K)), [n], [n]),
    ]
    res = milp(
        c=np.zeros(K),
        constraints=cons,
        integrality=np.ones(K),
        bounds=Bounds(0, n),
        options={"time_limit": time_limit, "presolve": True},
    )
    return res


# --------------------------------------------------------------------------- #
def check_true_config() -> None:
    print("Sanity check: the exact n=25 configuration satisfies (ARC).")
    rho = np.array([0.0] + [2 * math.cos(math.pi / 8)] * 8
                   + [2 + math.sqrt(2)] * 8 + [4 * math.cos(math.pi / 8)] * 8)
    worst = (np.inf, 0.0)
    for t in np.linspace(1e-4, R25, 40001):
        s = float(alpha(t, rho).sum())
        if s - math.pi < worst[0]:
            worst = (s - math.pi, t)
    print(f"  min over t of (sum alpha - pi) = {worst[0]:.3e} at t = {worst[1]:.6f}")
    for t, nm in ((1.0, "1"), (1 + math.sqrt(2), "1+sqrt2"),
                  (math.sqrt(5 + 2 * math.sqrt(2)), "tx"), (R25, "R")):
        s = float(alpha(t, rho).sum())
        print(f"  t = {nm:8s}: sum alpha = {math.degrees(s):9.5f} deg "
              f"(need >= 180)  slack = {math.degrees(s - math.pi):+9.5f} deg")


def scan(n_cells: int, n_t: int) -> None:
    print(f"\n{'='*78}")
    print(f"(ILP) feasibility scan   cells = {n_cells}, t-grid = {n_t}")
    print(f"{'='*78}")
    print(f"{'R':>10} {'LP feas':>9} {'ILP feas':>9}   {'verdict':<40}")
    for R in [4.181541, 4.20, 4.25, 4.30, 4.3622, 4.40]:
        _, _, A = build(R, n_cells, n_t)
        lp = lp_feasible(A)
        res = ilp_feasible(A)
        ok = bool(res.success)
        verdict = "" if ok else f"25 disks CANNOT cover disk({R}) => r(25) < {R}"
        print(f"{R:10.6f} {str(lp):>9} {str(ok):>9}   {verdict:<40}")


def bisect(n_cells: int, n_t: int, lo: float = 4.10, hi: float = 4.45,
           iters: int = 14) -> float:
    print(f"\nbisecting the (ILP) infeasibility threshold "
          f"(cells={n_cells}, t-grid={n_t})")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, _, A = build(mid, n_cells, n_t)
        res = ilp_feasible(A)
        if res.success:
            lo = mid          # feasible: no contradiction yet
        else:
            hi = mid          # infeasible: bound improves
        print(f"    R = {mid:.7f}  ILP feasible = {bool(res.success)}   "
              f"bracket [{lo:.7f}, {hi:.7f}]")
    return hi


def main() -> None:
    check_true_config()
    scan(n_cells=400, n_t=1200)
    b = bisect(n_cells=400, n_t=1200)
    print(f"\n  => rigorous bound from the integer radial-arc condition: "
          f"r(25) <= {b:.6f}")
    print(f"     (measure/LP relaxation only gives 4.362226)")


if __name__ == "__main__":
    main()
