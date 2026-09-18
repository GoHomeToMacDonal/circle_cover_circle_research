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


def _record_candidate(kind: str, point: np.ndarray, defining_sites: tuple[int, ...],
                      C: np.ndarray, R: float, tol: float) -> dict | None:
    point = np.asarray(point, dtype=float)
    radial = float(np.hypot(point[0], point[1]))
    if radial > R + tol:
        return None
    distances = np.linalg.norm(C - point[None, :], axis=1)
    value = float(distances.min())
    nearest = tuple(int(i) for i in np.flatnonzero(distances <= value + tol))
    if not set(defining_sites).issubset(nearest):
        return None
    return {
        "kind": kind,
        "point": [float(point[0]), float(point[1])],
        "defining_sites": [int(i) for i in defining_sites],
        "nearest_sites": list(nearest),
        "value": value,
        # Positive means outside S_R; boundary candidates should be near zero.
        "radial_residual": radial - float(R),
    }


def hole_candidates(C: np.ndarray, R: float, tol: float = 1e-9) -> list[dict]:
    """Enumerate legal floating-point deepest-hole candidates.

    The returned records are exploratory numerical observations, not an exact
    certificate.  A candidate is retained only when its point lies in the closed
    target disk; ``nearest_sites`` reports the actual numerical nearest sites,
    which need not equal ``defining_sites`` for a non-active geometric source.
    ``radial_residual`` is ``|point|-R``.
    """
    C = np.asarray(C, dtype=float)
    R = float(R)
    n = len(C)
    points: list[np.ndarray] = []
    kinds: list[str] = []
    definitions: list[tuple[int, ...]] = []

    def add_batch(kind: str, batch: np.ndarray, defs):
        batch = np.asarray(batch, dtype=float).reshape((-1, 2))
        points.extend(batch)
        kinds.extend([kind] * len(batch))
        definitions.extend([tuple(int(x) for x in item) for item in defs])

    distances0 = np.linalg.norm(C, axis=1)
    min0 = float(distances0.min()) if n else 0.0
    add_batch("origin", np.zeros((1, 2)),
              [tuple(np.flatnonzero(distances0 <= min0 + tol))])

    # All non-collinear triple circumcentres.
    if n >= 3:
        idx = _triples(n)
        a, b, c = C[idx[:, 0]], C[idx[:, 1]], C[idx[:, 2]]
        det = 2.0 * (a[:, 0] * (b[:, 1] - c[:, 1])
                     + b[:, 0] * (c[:, 1] - a[:, 1])
                     + c[:, 0] * (a[:, 1] - b[:, 1]))
        good = np.abs(det) > 1e-13
        a, b, c, det, idx = a[good], b[good], c[good], det[good], idx[good]
        aa, bb, cc = (a * a).sum(axis=1), (b * b).sum(axis=1), (c * c).sum(axis=1)
        triple_points = np.stack([
            (aa * (b[:, 1] - c[:, 1]) + bb * (c[:, 1] - a[:, 1])
             + cc * (a[:, 1] - b[:, 1])) / det,
            (aa * (c[:, 0] - b[:, 0]) + bb * (a[:, 0] - c[:, 0])
             + cc * (b[:, 0] - a[:, 0])) / det,
        ], axis=1)
        add_batch("triple-circumcenter", triple_points, idx)

    # Both intersections of every perpendicular bisector with S_R.
    if n >= 2:
        pair_i, pair_j = np.triu_indices(n, k=1)
        a, b = C[pair_i], C[pair_j]
        delta = b - a
        length = np.hypot(delta[:, 0], delta[:, 1])
        good = length > 1e-13
        a, b, delta, length = a[good], b[good], delta[good], length[good]
        pair_defs = np.stack([pair_i[good], pair_j[good]], axis=1)
        midpoint = 0.5 * (a + b)
        direction = np.stack([-delta[:, 1], delta[:, 0]], axis=1) / length[:, None]
        linear = (midpoint * direction).sum(axis=1)
        discriminant = linear * linear + R * R - (midpoint * midpoint).sum(axis=1)
        good = discriminant >= -tol
        midpoint, direction, linear, discriminant = (x[good] for x in
            (midpoint, direction, linear, discriminant))
        pair_defs = pair_defs[good]
        root = np.sqrt(np.maximum(discriminant, 0.0))
        add_batch("boundary-bisector",
                  np.concatenate((midpoint + (-linear + root)[:, None] * direction,
                                  midpoint + (-linear - root)[:, None] * direction)),
                  np.concatenate((pair_defs, pair_defs)))

    # Stationary points of the restriction to S_R for every non-origin site.
    rho = np.hypot(C[:, 0], C[:, 1])
    nonzero = rho > 1e-13
    if np.any(nonzero):
        indices = np.flatnonzero(nonzero)
        dirs = C[nonzero] / rho[nonzero, None]
        add_batch("boundary-stationary",
                  np.concatenate((R * dirs, -R * dirs)),
                  np.concatenate((indices[:, None], indices[:, None])))

    P = np.asarray(points, dtype=float)
    D = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)
    values = D.min(axis=1)
    nearest_mask = D <= values[:, None] + tol
    radial = np.hypot(P[:, 0], P[:, 1])
    legal = radial <= R + tol
    out: list[dict] = []
    for row, (kind, defining) in enumerate(zip(kinds, definitions)):
        if not legal[row]:
            continue
        out.append({
            "kind": kind,
            "point": [float(P[row, 0]), float(P[row, 1])],
            "defining_sites": list(defining),
            "nearest_sites": [int(i) for i in np.flatnonzero(nearest_mask[row])],
            "value": float(values[row]),
            "radial_residual": float(radial[row] - R),
            "status": "floating-point exploratory",
        })
    return out


def deepest_hole(C: np.ndarray, R: float) -> tuple[float, np.ndarray]:
    records = hole_candidates(C, R)
    if not records:
        return 0.0, np.zeros(2)
    k = int(np.argmax([record["value"] for record in records]))
    record = records[k]
    return float(record["value"]), np.asarray(record["point"], dtype=float)


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
