"""The exact n=25 configuration and its critical (tight) points.

Closed form.  With u = cos(pi/8) = sqrt(2+sqrt2)/2:

    c_0   = (0,0)
    ring A: 8 centres at rho_A = 2u          , angles pi/8 + k*pi/4
    ring B: 8 centres at rho_B = 4u^2 = 2+sqrt2, angles     k*pi/4
    ring C: 8 centres at rho_C = 4u          , angles pi/8 + k*pi/4

covers exactly the disk of radius R = sqrt6 + sqrt3 = sqrt3 (1+sqrt2).

Note the nesting: {c_0} u ring A is precisely the (proved-optimal) n=9
configuration, which covers the disk of radius 1+sqrt2, and rho_B - 1 = 1+sqrt2,
so the inner tips of ring B sit exactly on that inner disk's boundary.
"""

from __future__ import annotations

import math

import numpy as np

U = math.cos(math.pi / 8)
RHO_A = 2 * U
RHO_B = 2 + math.sqrt(2)
RHO_C = 4 * U
R25 = math.sqrt(6) + math.sqrt(3)
R9 = 1 + math.sqrt(2)


def centers25() -> np.ndarray:
    out = [np.zeros((1, 2))]
    for rho, th in ((RHO_A, math.pi / 8), (RHO_B, 0.0), (RHO_C, math.pi / 8)):
        a = th + np.arange(8) * math.pi / 4
        out.append(np.stack([rho * np.cos(a), rho * np.sin(a)], axis=1))
    return np.concatenate(out, axis=0)


LABELS = ["O"] + [f"A{k}" for k in range(8)] + [f"B{k}" for k in range(8)] + [
    f"C{k}" for k in range(8)
]


def tight_points(centers: np.ndarray, R: float, tol: float = 1e-9):
    """Points of the disk(R) that are covered with zero slack.

    These are exactly the points p with |p| <= R and min_i |p - c_i| = 1: the
    places where the covering is critical.  Any competing configuration that
    covers disk(R) must cover all of them, so they are the natural certificate
    set.  Candidates: pairwise circle intersections, near/far points of each
    circle, and points of S_R.
    """
    n = len(centers)
    cand = []
    i_idx, j_idx = np.triu_indices(n, k=1)
    a, b = centers[i_idx], centers[j_idx]
    d = b - a
    dist = np.hypot(d[:, 0], d[:, 1])
    ok = (dist > 1e-12) & (dist < 2.0)
    a, b, d, dist = a[ok], b[ok], d[ok], dist[ok]
    mid = a + 0.5 * d
    hh = np.sqrt(np.maximum(1.0 - 0.25 * dist**2, 0.0))
    perp = np.stack([-d[:, 1], d[:, 0]], axis=1) / dist[:, None]
    cand.append(mid + perp * hh[:, None])
    cand.append(mid - perp * hh[:, None])
    rho = np.hypot(centers[:, 0], centers[:, 1])
    nz = rho > 1e-12
    cand.append(centers[nz] * ((rho[nz] - 1) / rho[nz])[:, None])
    cand.append(centers[nz] * ((rho[nz] + 1) / rho[nz])[:, None])
    P = np.concatenate(cand, axis=0)
    dd = np.linalg.norm(P[:, None, :] - centers[None, :, :], axis=2)
    slack = 1.0 - dd.min(axis=1)          # >= 0 means covered; 0 means tight
    rad = np.hypot(P[:, 0], P[:, 1])
    keep = (rad <= R + tol) & (np.abs(slack) <= tol)
    return P[keep], rad[keep], dd[keep]


def dedup(P: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    out: list[np.ndarray] = []
    for p in P:
        if all(np.hypot(*(p - q)) > tol for q in out):
            out.append(p)
    return np.array(out)


def main() -> None:
    import sys

    sys.path.insert(0, "src")
    from geom import circle_gap, covering_radius

    C = centers25()
    print(f"rho_A = {RHO_A:.15f}   (= 2 cos(pi/8))")
    print(f"rho_B = {RHO_B:.15f}   (= 2 + sqrt2)")
    print(f"rho_C = {RHO_C:.15f}   (= 4 cos(pi/8))")
    Rc = covering_radius(C)
    print(f"\ncovering radius   = {Rc:.15f}")
    print(f"sqrt6 + sqrt3     = {R25:.15f}")
    print(f"difference        = {Rc - R25:.3e}")

    P, rad, dd = tight_points(C, R25)
    P = dedup(P)
    print(f"\ntight points inside disk(R): {len(P)}")
    rr = np.hypot(P[:, 0], P[:, 1])
    th = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360
    order = np.lexsort((th, np.round(rr, 6)))
    for i in order:
        d = np.linalg.norm(P[i] - C, axis=1)
        owners = [LABELS[j] for j in np.argsort(d)[:4] if d[j] < 1 + 1e-7]
        print(f"  |p|={rr[i]:9.6f}  arg={th[i]:8.3f}  on circles {owners}")

    # radial profile of the largest angular gap: where is the covering tight?
    print("\nradial profile of max angular gap (deg), 0 = fully covered:")
    for t in np.linspace(0.02, R25, 60):
        g = math.degrees(circle_gap(C, t))
        bar = "#" * int(min(g, 20) * 2)
        print(f"  t={t:7.4f}  gap={g:8.4f} {bar}")


if __name__ == "__main__":
    main()
