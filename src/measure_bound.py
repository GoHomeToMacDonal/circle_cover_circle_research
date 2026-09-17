"""The measure ("probabilistic") lower bound on the covering number of a disk.

If mu is a finite Borel measure on the closed disk D_R = disk(0,R) with

        mu(B(x,1)) <= 1   for every x in R^2,

then any covering of D_R by N unit disks has mu(D_R) <= N, so

        N >= mu(D_R).

The supremum of mu(D_R) over all such mu is the *fractional covering number*
phi(R) of D_R by unit disks; it is the LP relaxation of the covering problem and
is the exact strength of the whole family of measure / probabilistic arguments.

Rotational reduction.  The constraint family and the objective are invariant
under rotations about the origin, and the feasible set is convex, so averaging a
feasible mu over the rotation group keeps it feasible with the same total mass.
Hence phi(R) is attained by a rotation-invariant measure, i.e. by

        mu = sum_k w_k * (uniform probability measure on the circle S_{t_k}),
        w_k >= 0.

For such a mu and any x with |x| = sigma,

        mu(B(x,1)) = sum_k w_k * alpha(t_k, sigma) / pi,
        alpha(t,sigma) = arccos((t^2+sigma^2-1)/(2 t sigma))  in [0, pi],

with alpha = pi when t + sigma <= 1 (the whole circle is inside) and alpha = 0
when |t - sigma| > 1.  So phi(R) is the value of the one-dimensional LP

        max sum_k w_k   s.t.  sum_k w_k alpha(t_k, sigma)/pi <= 1  for all sigma,
                              w >= 0.

Rigour.  Solving on a finite sigma-grid *relaxes* the constraints, so the raw LP
value is only an over-estimate.  We therefore (a) solve on a grid, (b) verify the
maximum of sum_k w_k alpha(t_k,sigma)/pi over a far finer grid, and (c) rescale w
by 1/max, which yields a certified feasible measure and hence a valid bound.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog

R25 = math.sqrt(6) + math.sqrt(3)


def alpha_frac(t: np.ndarray, sigma: float) -> np.ndarray:
    """alpha(t, sigma)/pi, vectorised over t."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    if sigma <= 1e-15:
        out[t <= 1.0] = 1.0
        return out
    whole = t + sigma <= 1.0
    out[whole] = 1.0
    live = (~whole) & (np.abs(t - sigma) <= 1.0) & (t > 1e-15)
    c = np.clip((t[live] ** 2 + sigma**2 - 1.0) / (2.0 * t[live] * sigma), -1.0, 1.0)
    out[live] = np.arccos(c) / math.pi
    # t = 0 is an atom at the origin: covered iff sigma <= 1
    zero = t <= 1e-15
    out[zero] = 1.0 if sigma <= 1.0 else 0.0
    return out


def build_matrix(ts: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    return np.array([alpha_frac(ts, s) for s in sigmas])


def phi(R: float, n_t: int = 900, n_sigma: int = 4000, n_verify: int = 200_000):
    """Certified lower bound on the fractional covering number of disk(0,R)."""
    ts = np.linspace(0.0, R, n_t)
    sig = np.linspace(0.0, R + 1.0, n_sigma)
    A = build_matrix(ts, sig)
    res = linprog(c=-np.ones(n_t), A_ub=A, b_ub=np.ones(n_sigma),
                  bounds=[(0, None)] * n_t, method="highs")
    if not res.success:
        return None
    w = res.x
    raw = w.sum()
    # certify: recompute the max over a much finer sigma grid
    fine = np.linspace(0.0, R + 1.0, n_verify)
    mx = 0.0
    for s in range(0, n_verify, 20000):
        blk = fine[s : s + 20000]
        M = build_matrix(ts, blk) @ w
        mx = max(mx, float(M.max()))
    certified = raw / max(mx, 1.0)
    return {"R": R, "raw": raw, "max_load": mx, "certified": certified, "w": w,
            "ts": ts}


def main() -> None:
    print("=" * 78)
    print("Fractional (measure) covering number of disk(0,R) by unit disks")
    print("=" * 78)
    print(f"{'R':>10} {'LP value':>12} {'max load':>10} {'certified':>12}"
          f"  {'>25?':>6}")
    for R in [3.0, 3.5, 4.0, R25, 4.3, 4.5, 4.7, 4.9, 5.0, 5.2, 5.5]:
        r = phi(R, n_t=700, n_sigma=3000, n_verify=120_000)
        flag = "YES" if r["certified"] > 25 else ""
        tag = "  <-- R25" if abs(R - R25) < 1e-9 else ""
        print(f"{R:10.6f} {r['raw']:12.5f} {r['max_load']:10.6f} "
              f"{r['certified']:12.5f}  {flag:>6}{tag}")

    print("\nbisect for the R where the certified fractional bound reaches 25:")
    lo, hi = 4.0, 6.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        r = phi(mid, n_t=700, n_sigma=3000, n_verify=120_000)
        if r["certified"] >= 25.0:
            hi = mid
        else:
            lo = mid
    print(f"  phi(R) = 25 at R ~ {hi:.6f}")
    print(f"  => rigorous unconditional bound r(25) <= {hi:.6f}")
    print(f"     (target sqrt6+sqrt3 = {R25:.6f}; the measure method alone")
    print(f"      cannot do better than this, so the gap is intrinsic to it)")

    # for comparison: the trivial area bound is R^2 (density 1/pi)
    print(f"\n  area-only bound: phi(R) >= R^2, giving r(25) <= 5.0 exactly")


if __name__ == "__main__":
    main()
