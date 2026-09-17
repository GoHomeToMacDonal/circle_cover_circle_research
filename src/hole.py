"""Exact "deepest hole" of a configuration inside disk(0,R).

    H(C, R) = max_{|p| <= R} min_i |p - c_i|

so disk(0,R) is covered by unit disks at C iff H(C,R) <= 1.  The maximum of the
distance-to-nearest-site function over a convex region is attained at
  * a vertex of the Voronoi diagram of C lying inside the region
    (= circumcentre of some triple of sites), or
  * a point of the boundary circle S_R, where the restriction of the function has
    its local maxima either at a crossing of S_R with a perpendicular bisector of
    two sites, or at a point of S_R stationary for a single site, i.e. at
    +- R c_i / |c_i|.
Enumerating those O(n^3) candidates and keeping the feasible ones gives H exactly.
"""

from __future__ import annotations

import itertools

import numpy as np


_TRIPLE_CACHE: dict[int, np.ndarray] = {}


def _triples(n: int) -> np.ndarray:
    if n not in _TRIPLE_CACHE:
        _TRIPLE_CACHE[n] = np.array(list(itertools.combinations(range(n), 3)))
    return _TRIPLE_CACHE[n]


def _circumcenters(C: np.ndarray) -> np.ndarray:
    n = len(C)
    if n < 3:
        return np.zeros((0, 2))
    idx = _triples(n)
    a, b, c = C[idx[:, 0]], C[idx[:, 1]], C[idx[:, 2]]
    d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1])
             + c[:, 0] * (a[:, 1] - b[:, 1]))
    ok = np.abs(d) > 1e-13
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    aa = (a * a).sum(axis=1)
    bb = (b * b).sum(axis=1)
    cc = (c * c).sum(axis=1)
    ux = (aa * (b[:, 1] - c[:, 1]) + bb * (c[:, 1] - a[:, 1])
          + cc * (a[:, 1] - b[:, 1])) / d
    uy = (aa * (c[:, 0] - b[:, 0]) + bb * (a[:, 0] - c[:, 0])
          + cc * (b[:, 0] - a[:, 0])) / d
    return np.stack([ux, uy], axis=1)


def _bisector_circle_points(C: np.ndarray, R: float) -> np.ndarray:
    """Intersections of perpendicular bisectors of pairs with the circle S_R."""
    n = len(C)
    i, j = np.triu_indices(n, k=1)
    a, b = C[i], C[j]
    m = 0.5 * (a + b)
    d = b - a
    L = np.hypot(d[:, 0], d[:, 1])
    ok = L > 1e-13
    m, d, L = m[ok], d[ok], L[ok]
    u = np.stack([-d[:, 1], d[:, 0]], axis=1) / L[:, None]   # direction of bisector
    # |m + s u| = R  ->  s^2 + 2 s (m.u) + |m|^2 - R^2 = 0
    b1 = (m * u).sum(axis=1)
    cq = (m * m).sum(axis=1) - R * R
    disc = b1 * b1 - cq
    good = disc >= 0
    m, u, b1, disc = m[good], u[good], b1[good], disc[good]
    sq = np.sqrt(disc)
    return np.concatenate([m + (-b1 + sq)[:, None] * u,
                           m + (-b1 - sq)[:, None] * u], axis=0)


def deepest_hole(C: np.ndarray, R: float) -> tuple[float, np.ndarray]:
    C = np.asarray(C, dtype=float)
    cand = [np.zeros((1, 2))]
    v = _circumcenters(C)
    if len(v):
        cand.append(v[np.hypot(v[:, 0], v[:, 1]) <= R + 1e-12])
    cand.append(_bisector_circle_points(C, R))
    rho = np.hypot(C[:, 0], C[:, 1])
    nz = rho > 1e-13
    if nz.any():
        dirs = C[nz] / rho[nz][:, None]
        cand.append(R * dirs)
        cand.append(-R * dirs)
    P = np.concatenate([c for c in cand if len(c)], axis=0)
    keep = np.hypot(P[:, 0], P[:, 1]) <= R + 1e-9
    P = P[keep]
    if len(P) == 0:
        return 0.0, np.zeros(2)
    d = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2).min(axis=1)
    k = int(np.argmax(d))
    return float(d[k]), P[k]


def covers(C: np.ndarray, R: float, tol: float = 1e-12) -> bool:
    return deepest_hole(C, R)[0] <= 1.0 + tol


def deepest_hole_symmetric(C: np.ndarray, R: float, k: int) -> float:
    """Same as deepest_hole but for a configuration invariant under rotation by
    2 pi / k: the maximum is attained in the wedge 0 <= arg p <= 2 pi / k, so only
    candidates in that wedge need their distances evaluated."""
    C = np.asarray(C, dtype=float)
    cand = [np.zeros((1, 2))]
    v = _circumcenters(C)
    if len(v):
        cand.append(v)
    cand.append(_bisector_circle_points(C, R))
    rho = np.hypot(C[:, 0], C[:, 1])
    nz = rho > 1e-13
    if nz.any():
        dirs = C[nz] / rho[nz][:, None]
        cand.append(R * dirs)
        cand.append(-R * dirs)
    P = np.concatenate([c for c in cand if len(c)], axis=0)
    rr = np.hypot(P[:, 0], P[:, 1])
    ang = np.arctan2(P[:, 1], P[:, 0]) % (2 * np.pi)
    wedge = 2 * np.pi / k
    keep = (rr <= R + 1e-9) & ((ang <= wedge + 1e-9) | (rr <= 1e-9))
    P = P[keep]
    if len(P) == 0:
        return 0.0
    d = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2).min(axis=1)
    return float(d.max())
