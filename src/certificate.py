"""Finite certificates for upper bounds on r(25).

THE REDUCTION.  Suppose 25 unit disks cover disk(0, R') with R' > R* = sqrt6+sqrt3.
Scaling by R*/R' shows that 25 disks of radius rho = R*/R' < 1 cover disk(0, R*),
hence cover every finite subset P of disk(0, R*).  Contrapositive:

    if some finite P subset disk(0,R*) cannot be covered by 25 disks of
    radius rho, then  r(25) <= R*/rho.

Two certificates for "P needs more than 25 disks of radius rho":

(1) INTEGRAL (set cover).  The minimum number of radius-rho disks covering P is
    the optimum of a set-cover ILP.  A complete finite candidate set of disks is
    obtained from the classical locking argument: translate a covering disk until
    two points of P lie on its boundary; hence it suffices to take centres at
    - every point of P                            (disks containing <= 1 point)
    - every intersection of two circles of radius rho about points of P.
    Every subset of P cut out by a radius-rho disk is contained in one of the
    resulting candidate subsets, so the ILP optimum is exact.

(2) FRACTIONAL (measure / "probabilistic" method).  The LP relaxation dual is:
    find weights w_p >= 0 with sum_{p in D} w_p <= 1 for every radius-rho disk D
    and sum_p w_p > 25.  Such a w is a hand-checkable certificate: no disk can
    carry more than one unit of weight, so 25 disks carry at most 25, less than
    the total.  This is exactly the measure-theoretic lower bound on the covering
    number, and its optimum equals the LP relaxation value of (1).

ADAPTIVE POINT GENERATION.  Start from the 48 tight points.  Solve (1); if it
finds <= 25 disks of radius rho covering P, those disks necessarily fail to cover
all of disk(0,R*); locate a deepest uncovered point, add its D_8 orbit to P, and
repeat.  Every iteration strictly strengthens P.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix

from config25 import R25, centers25, dedup, tight_points

EPS = 1e-9


# --------------------------------------------------------------------------- #
# candidate disks
# --------------------------------------------------------------------------- #
def candidate_centers(P: np.ndarray, rho: float) -> np.ndarray:
    """Complete set of candidate centres for maximal radius-rho covers of P."""
    out = [P.copy()]
    m = len(P)
    i, j = np.triu_indices(m, k=1)
    a, b = P[i], P[j]
    d = b - a
    dist = np.hypot(d[:, 0], d[:, 1])
    ok = (dist > EPS) & (dist < 2 * rho - 1e-13)
    a, d, dist = a[ok], d[ok], dist[ok]
    mid = a + 0.5 * d
    h = np.sqrt(np.maximum(rho**2 - 0.25 * dist**2, 0.0))
    perp = np.stack([-d[:, 1], d[:, 0]], axis=1) / dist[:, None]
    out.append(mid + perp * h[:, None])
    out.append(mid - perp * h[:, None])
    return np.concatenate(out, axis=0)


def candidate_subsets(P: np.ndarray, rho: float, chunk: int = 2000):
    """Distinct maximal subsets P n B(x,rho) over the candidate centres x."""
    X = candidate_centers(P, rho)
    m = len(P)
    seen: dict[bytes, np.ndarray] = {}
    for s in range(0, len(X), chunk):
        blk = X[s : s + chunk]
        dd = np.linalg.norm(blk[:, None, :] - P[None, :, :], axis=2)
        inside = dd <= rho + 1e-12
        for row in inside:
            if not row.any():
                continue
            seen[np.packbits(row).tobytes()] = row
    subs = list(seen.values())
    # drop non-maximal subsets
    keep = []
    arr = np.array(subs)
    order = np.argsort(-arr.sum(axis=1))
    arr = arr[order]
    for k in range(len(arr)):
        r = arr[k]
        dominated = False
        for kk in keep:
            if np.all(r <= arr[kk]):
                dominated = True
                break
        if not dominated:
            keep.append(k)
    return arr[keep]


# --------------------------------------------------------------------------- #
# certificates
# --------------------------------------------------------------------------- #
def fractional_bound(subs: np.ndarray, verbose: bool = False):
    """LP: max sum w_p  s.t. sum_{p in S} w_p <= 1 for all candidate S, w >= 0."""
    m = subs.shape[1]
    res = linprog(
        c=-np.ones(m),
        A_ub=csc_matrix(subs.astype(float)),
        b_ub=np.ones(len(subs)),
        bounds=[(0, None)] * m,
        method="highs",
    )
    if not res.success:
        return None, None
    return -res.fun, res.x


def integral_cover(subs: np.ndarray, time_limit: float = 300.0):
    """ILP: minimum number of candidate disks covering all points."""
    nS, m = subs.shape
    A = csc_matrix(subs.T.astype(float))       # rows = points, cols = disks
    res = milp(
        c=np.ones(nS),
        constraints=LinearConstraint(A, np.ones(m), np.full(m, np.inf)),
        integrality=np.ones(nS),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "presolve": True},
    )
    return res


# --------------------------------------------------------------------------- #
# find where a candidate covering fails
# --------------------------------------------------------------------------- #
def deepest_uncovered(cent: np.ndarray, rho: float, R: float, n_r: int = 700,
                      n_a: int = 2400) -> tuple[float, np.ndarray]:
    """Point of disk(0,R) maximising min_i |p - c_i| - rho (polar grid + refine)."""
    rr = np.linspace(0.0, R, n_r)
    aa = np.linspace(0.0, 2 * math.pi, n_a, endpoint=False)
    best = (-np.inf, np.zeros(2))
    for r in rr:
        pts = np.stack([r * np.cos(aa), r * np.sin(aa)], axis=1)
        d = np.linalg.norm(pts[:, None, :] - cent[None, :, :], axis=2).min(axis=1)
        k = int(np.argmax(d))
        if d[k] - rho > best[0]:
            best = (float(d[k] - rho), pts[k])
    # local refine
    p = best[1].copy()
    step = R / n_r
    val = best[0]
    for _ in range(60):
        improved = False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = p + step * np.array([dx, dy])
            if np.hypot(*q) > R:
                q = q * (R / np.hypot(*q))
            v = float(np.linalg.norm(q - cent, axis=1).min() - rho)
            if v > val + 1e-15:
                val, p, improved = v, q, True
        if not improved:
            step *= 0.5
            if step < 1e-13:
                break
    return val, p


def d8_orbit(p: np.ndarray) -> np.ndarray:
    """The D_8 orbit (order 16) of a point."""
    out = []
    for k in range(8):
        a = k * math.pi / 4
        c, s = math.cos(a), math.sin(a)
        for q in (p, np.array([p[0], -p[1]])):
            out.append(np.array([c * q[0] - s * q[1], s * q[0] + c * q[1]]))
    return dedup(np.array(out), tol=1e-9)


# --------------------------------------------------------------------------- #
def run(rho: float, rounds: int = 12, seed_extra: int = 0, verbose: bool = True):
    C = centers25()
    T, _, _ = tight_points(C, R25, tol=1e-9)
    P = dedup(T, tol=1e-7)
    if seed_extra:
        # add a coarse D_8-symmetric net for a stronger start
        extra = []
        for r in np.linspace(0.35, R25, seed_extra):
            for a in np.linspace(0, math.pi / 8, max(2, int(seed_extra / 3))):
                extra.append([r * math.cos(a), r * math.sin(a)])
        for q in np.array(extra):
            P = np.concatenate([P, d8_orbit(q)], axis=0)
        P = dedup(P, tol=1e-6)

    print(f"\n{'='*76}\nrho = {rho:.6f}   =>  a certificate would give r(25) <= "
          f"{R25/rho:.9f}\n{'='*76}")
    for it in range(rounds):
        t0 = time.time()
        subs = candidate_subsets(P, rho)
        lp, w = fractional_bound(subs)
        res = integral_cover(subs)
        k = int(round(res.fun)) if res.success else None
        print(f"  round {it:2d}: |P| = {len(P):4d}  candidate disks = {len(subs):6d}  "
              f"LP = {lp:8.4f}  ILP = {k}   ({time.time()-t0:.1f}s)")
        if k is None:
            print("     ILP failed"); break
        if k >= 26:
            print(f"\n  *** CERTIFICATE: P needs {k} disks of radius {rho} ***")
            print(f"      => r(25) <= {R25/rho:.9f}")
            return {"proved": True, "rho": rho, "bound": R25 / rho, "P": P,
                    "k": k, "lp": lp}
        # extract the chosen disks and find an uncovered point of disk(R*)
        X = candidate_centers(P, rho)
        chosen = np.nonzero(res.x > 0.5)[0]
        cent = fit_centers(P, subs[chosen], rho)
        depth, q = deepest_uncovered(cent, rho, R25)
        print(f"           deepest uncovered depth = {depth:.6f} at "
              f"|q|={np.hypot(*q):.5f} arg={math.degrees(math.atan2(q[1],q[0])):.3f}")
        if depth <= 1e-9:
            print("     the chosen disks really do cover disk(R*): rho is too small")
            return {"proved": False, "rho": rho, "P": P, "k": k, "lp": lp}
        P = dedup(np.concatenate([P, d8_orbit(q)], axis=0), tol=1e-7)
    return {"proved": False, "rho": rho, "P": P, "lp": lp}


def fit_centers(P: np.ndarray, subs: np.ndarray, rho: float) -> np.ndarray:
    """A concrete centre for each chosen subset (min enclosing circle centre)."""
    out = []
    for row in subs:
        pts = P[row.astype(bool)]
        out.append(mec_center(pts))
    return np.array(out)


def mec_center(pts: np.ndarray) -> np.ndarray:
    """Minimum enclosing circle centre (small sets: brute force)."""
    n = len(pts)
    if n == 1:
        return pts[0]
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            c = 0.5 * (pts[i] + pts[j])
            r = np.linalg.norm(pts - c, axis=1).max()
            if best is None or r < best[0] - 1e-15:
                best = (r, c)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c = circumcenter(pts[i], pts[j], pts[k])
                if c is None:
                    continue
                r = np.linalg.norm(pts - c, axis=1).max()
                if r < best[0] - 1e-15:
                    best = (r, c)
    return best[1]


def circumcenter(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-14:
        return None
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    return np.array([ux, uy])


def main() -> None:
    import sys

    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 0.93
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed_extra = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    run(rho, rounds=rounds, seed_extra=seed_extra)


if __name__ == "__main__":
    main()
