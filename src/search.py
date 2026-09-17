"""Unrestricted numerical search for coverings of disk(0,R) by 25 unit disks.

For fixed R the question "can 25 unit disks cover disk(0,R)?" is the continuous
25-centre problem

    minimise   H(C, R) = max_{|p| <= R} min_i |p - c_i|
    over       C in (R^2)^25,

and 25 unit disks cover disk(0,R) exactly when the minimum is <= 1.

Algorithm (cutting planes + Chebyshev alternation):
  * keep a finite working set P of points of disk(0,R);
  * alternate: assign each p in P to its nearest centre, then move each centre to
    the centre of the minimum enclosing circle of its cluster (the exact optimum
    for one cluster).  This never increases max_{p in P} min_i |p - c_i|;
  * evaluate H(C,R) EXACTLY with hole.deepest_hole (Voronoi vertices and boundary
    stationary points -- no discretisation), and add the maximiser to P;
  * repeat.  P grows only where it matters, so the value converges to the true
    continuous optimum rather than to a net artefact.
"""

from __future__ import annotations

import math

import numpy as np

from config25 import R25, centers25
from geom import covering_radius
from hole import deepest_hole


def mec(pts: np.ndarray, iters: int = 150) -> np.ndarray:
    """Approximate minimum enclosing circle centre (Badoiu-Clarkson)."""
    if len(pts) == 1:
        return pts[0].copy()
    c = pts.mean(axis=0)
    for k in range(1, iters + 1):
        d = np.linalg.norm(pts - c, axis=1)
        c = c + (pts[int(np.argmax(d))] - c) / (k + 1)
    return c


def polar_net(R: float, n_r: int = 26) -> np.ndarray:
    pts = [np.zeros((1, 2))]
    for k in range(1, n_r + 1):
        r = R * k / n_r
        m = max(6, int(round(2 * math.pi * r / (R / n_r))))
        a = np.linspace(0, 2 * math.pi, m, endpoint=False)
        pts.append(np.stack([r * np.cos(a), r * np.sin(a)], axis=1))
    return np.concatenate(pts, axis=0)


def alternate(P: np.ndarray, C: np.ndarray, iters: int = 40) -> np.ndarray:
    for _ in range(iters):
        d = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)
        lab = np.argmin(d, axis=1)
        newC = C.copy()
        for i in range(len(C)):
            cl = P[lab == i]
            if len(cl):
                newC[i] = mec(cl)
            else:
                newC[i] = P[int(np.argmax(d.min(axis=1)))]
        if np.allclose(newC, C, atol=1e-14):
            return newC
        C = newC
    return C


def polish(C: np.ndarray, R: float, rng: np.random.Generator,
           tries: int = 400) -> tuple[float, np.ndarray]:
    """Randomised local descent directly on the exact objective H(C,R)."""
    h = deepest_hole(C, R)[0]
    step = 0.02
    while step > 1e-11:
        improved = False
        for _ in range(tries):
            i = int(rng.integers(len(C)))
            D = C.copy()
            D[i] = D[i] + rng.normal(scale=step, size=2)
            hh = deepest_hole(D, R)[0]
            if hh < h - 1e-15:
                h, C, improved = hh, D, True
        if not improved:
            step *= 0.4
    return h, C


def minimise_H(R: float, C0: np.ndarray, rounds: int = 25, n_r: int = 20):
    """Cutting-plane / alternation loop.  Always keeps the best exact H seen."""
    P = polar_net(R, n_r)
    C = C0.astype(float).copy()
    h0, q0 = deepest_hole(C, R)
    best = (h0, C.copy())
    P = np.vstack([P, q0])
    for _ in range(rounds):
        C = alternate(P, C)
        h, q = deepest_hole(C, R)
        if h < best[0]:
            best = (h, C.copy())
        P = np.vstack([P, q])
    return best


def random_start(rng: np.random.Generator, R: float) -> np.ndarray:
    if rng.random() < 0.45:
        a = rng.uniform(0, 2 * math.pi, 25)
        rr = R * np.sqrt(rng.uniform(0, 1, 25))
        return np.stack([rr * np.cos(a), rr * np.sin(a)], axis=1)
    k = int(rng.integers(3, 10))
    parts, left = [], 25
    if rng.random() < 0.7:
        parts.append(np.zeros((1, 2)))
        left -= 1
    while left >= k:
        rho = rng.uniform(0.25, R)
        ph = rng.uniform(0, 2 * math.pi)
        a = ph + np.arange(k) * 2 * math.pi / k
        parts.append(np.stack([rho * np.cos(a), rho * np.sin(a)], axis=1))
        left -= k
    if left:
        a = rng.uniform(0, 2 * math.pi, left)
        rr = R * np.sqrt(rng.uniform(0, 1, left))
        parts.append(np.stack([rr * np.cos(a), rr * np.sin(a)], axis=1))
    return np.concatenate(parts, axis=0)[:25]


def search(R: float, restarts: int, seed: int, seed_config=None, n_r: int = 20,
           rounds: int = 25, do_polish: bool = True):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    starts = [] if seed_config is None else [np.asarray(seed_config, dtype=float)]
    starts += [random_start(rng, R) for _ in range(restarts)]
    for C0 in starts:
        h, C = minimise_H(R, C0, rounds=rounds, n_r=n_r)
        if do_polish and h < best[0] + 0.02:
            h, C = polish(C, R, rng, tries=250)
        if h < best[0]:
            best = (h, C)
    return best


def main() -> None:
    import sys

    print("=" * 78)
    print("Unrestricted search: can 25 unit disks cover more than sqrt6+sqrt3?")
    print("=" * 78)
    C0 = centers25()
    h0, _ = deepest_hole(C0, R25)
    print(f"reference configuration: H(C*,R*) = {h0:.15f}  (must be exactly 1)")
    print(f"                         exact covering radius = {covering_radius(C0):.12f}")

    restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    print(f"\n{restarts} random restarts per R, plus the scaled reference config.")
    print(f"\n{'R':>10} {'min H found':>14}  {'covers?':>8}  {'note'}")
    for R in (R25, 4.1816, 4.185, 4.19, 4.20, 4.25):
        h, C = search(R, restarts=restarts, seed=11, seed_config=C0 * (R / R25))
        cov = h <= 1.0 + 1e-11
        note = ""
        if cov and R > R25 + 1e-9:
            note = "*** BEATS sqrt6+sqrt3 ***"
            np.save(f"data/beat_{R}.npy", C)
        print(f"{R:10.6f} {h:14.9f}  {str(cov):>8}  {note}")

    print("\nHow close does an unconstrained search get at R = sqrt6+sqrt3?")
    h, C = search(R25, restarts=restarts * 3, seed=99)
    print(f"  best H over restarts (no seeding) = {h:.9f}")
    print(f"  exact covering radius of that configuration = {covering_radius(C):.9f}")
    np.save("data/search_best.npy", C)


if __name__ == "__main__":
    main()
