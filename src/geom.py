"""Core geometry for the "circles covering circles" problem.

A configuration is a list of centres c_1..c_n in R^2; every covering circle has
unit radius.  Write U = union of the closed unit disks.  The *covering radius*

    R(C) = sup { R : disk(0,R) subset of U }

Key reduction.  disk(0,R) is contained in U iff for every t in [0,R] the whole
circle S_t = {|p| = t} is contained in U.  On S_t the disk B(c_i,1) cuts out a
single closed arc (empty, a point, a proper arc, or all of S_t), so "is S_t
covered" is a one-dimensional arc-covering question that can be decided exactly.

The combinatorial type of that arc system changes only at finitely many radii:

  * t = |rho_i - 1| and t = rho_i + 1   (an arc appears / vanishes)
  * t = |p| for p an intersection point of two of the unit circles
    (two arc endpoints cross)
  * t = 0

Between consecutive critical radii the covered/uncovered status of S_t is
constant, so we test one interior sample per gap and take

    R(C) = the smallest critical radius t* such that S_t is uncovered
           for t slightly larger than t*.

This is exact up to floating point and needs no derivative reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOL = 1e-12


# --------------------------------------------------------------------------- #
# arc covering of a single circle S_t
# --------------------------------------------------------------------------- #
def arcs_on_circle(centers: np.ndarray, t: float) -> list[tuple[float, float]]:
    """Angular intervals of S_t covered by the unit disks.

    Returns a list of (lo, hi) with lo <= hi (possibly hi > 2*pi), or the single
    interval (0, 2*pi) if some disk swallows S_t entirely.
    """
    rho = np.hypot(centers[:, 0], centers[:, 1])
    if t <= TOL:
        return [(0.0, 2 * np.pi)] if np.any(rho <= 1.0) else []
    out: list[tuple[float, float]] = []
    with np.errstate(invalid="ignore", divide="ignore"):
        cosv = (t * t + rho * rho - 1.0) / (2.0 * t * rho)
    for i in range(len(centers)):
        if rho[i] <= TOL:
            # concentric: covers all of S_t iff t <= 1
            if t <= 1.0:
                return [(0.0, 2 * np.pi)]
            continue
        if rho[i] + 1.0 < t or t < rho[i] - 1.0:
            continue
        if t + rho[i] <= 1.0:  # S_t entirely inside this disk
            return [(0.0, 2 * np.pi)]
        c = cosv[i]
        if c >= 1.0:
            continue
        half = np.arccos(max(c, -1.0))
        mid = np.arctan2(centers[i, 1], centers[i, 0])
        out.append((mid - half, mid + half))
    return out


def circle_gap(centers: np.ndarray, t: float) -> float:
    """Largest uncovered angular gap on S_t (0.0 means fully covered)."""
    arcs = arcs_on_circle(centers, t)
    if not arcs:
        return 2 * np.pi
    if any(hi - lo >= 2 * np.pi - TOL for lo, hi in arcs):
        return 0.0
    seg: list[tuple[float, float]] = []
    for lo, hi in arcs:
        span = hi - lo
        a = lo % (2 * np.pi)
        b = a + span
        if b <= 2 * np.pi:
            seg.append((a, b))
        else:
            seg.append((a, 2 * np.pi))
            seg.append((0.0, b - 2 * np.pi))
    seg.sort()
    lead = seg[0][0]  # uncovered piece before the first arc
    reach = seg[0][1]
    worst = 0.0
    for a, b in seg[1:]:
        if a > reach:
            worst = max(worst, a - reach)
            reach = b
        else:
            reach = max(reach, b)
    worst = max(worst, lead + max(2 * np.pi - reach, 0.0))
    return worst


def circle_covered(centers: np.ndarray, t: float, eps: float = 1e-11) -> bool:
    return circle_gap(centers, t) <= eps


# --------------------------------------------------------------------------- #
# critical radii
# --------------------------------------------------------------------------- #
def critical_radii(centers: np.ndarray) -> np.ndarray:
    rho = np.hypot(centers[:, 0], centers[:, 1])
    cand = [0.0]
    cand += list(np.abs(rho - 1.0))
    cand += list(rho + 1.0)
    n = len(centers)
    i_idx, j_idx = np.triu_indices(n, k=1)
    a, b = centers[i_idx], centers[j_idx]
    d = b - a
    dist = np.hypot(d[:, 0], d[:, 1])
    ok = (dist > TOL) & (dist < 2.0)
    a, b, d, dist = a[ok], b[ok], d[ok], dist[ok]
    if len(a):
        mid = a + d * 0.5
        hh = np.sqrt(np.maximum(1.0 - 0.25 * dist**2, 0.0))
        perp = np.stack([-d[:, 1], d[:, 0]], axis=1) / dist[:, None]
        for p in (mid + perp * hh[:, None], mid - perp * hh[:, None]):
            cand += list(np.hypot(p[:, 0], p[:, 1]))
    return np.unique(np.round(np.array(cand), 12))


def covering_radius(centers: np.ndarray) -> float:
    """Exact covering radius of the configuration.

    Walks the sorted critical radii; the answer is the left endpoint of the first
    gap (t_k, t_{k+1}) on which S_t is not covered.  Critical radii themselves are
    never sampled: at a tangency the angular gap behaves like sqrt(t - t*), so
    floating-point noise there is amplified to ~1e-6 and unusable.
    """
    centers = np.asarray(centers, dtype=float)
    if not circle_covered(centers, 0.0):
        return 0.0
    crit = critical_radii(centers)
    crit = crit[crit >= 0.0]
    for k in range(len(crit)):
        a = float(crit[k])
        b = float(crit[k + 1]) if k + 1 < len(crit) else a + 1.0
        if b <= a + 1e-14:
            continue
        for f in (0.5, 0.15, 0.85, 0.35, 0.65):
            if not circle_covered(centers, a + f * (b - a)):
                return a
    return float(crit[-1])


def worst_point(centers: np.ndarray, t: float) -> np.ndarray:
    """A point on S_t in the middle of the largest uncovered gap."""
    arcs = arcs_on_circle(centers, t)
    best = (-1.0, 0.0)
    for ang in np.linspace(0, 2 * np.pi, 20001):
        p = np.array([t * np.cos(ang), t * np.sin(ang)])
        d = np.linalg.norm(p - centers, axis=1).min()
        if d > best[0]:
            best = (d, ang)
    a = best[1]
    return np.array([t * np.cos(a), t * np.sin(a)])


def uncovered_depth(centers: np.ndarray, R: float, m: int = 400) -> float:
    """max over p in disk(R) of (min_i |p-c_i| - 1); <= 0 iff disk(R) covered."""
    g = np.linspace(-R, R, m)
    X, Y = np.meshgrid(g, g)
    inside = X**2 + Y**2 <= R**2
    P = np.stack([X[inside], Y[inside]], axis=1)
    d = np.linalg.norm(P[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
    return float(d.max() - 1.0)


# --------------------------------------------------------------------------- #
# symmetric parametrisations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RingConfig:
    """k-fold symmetric configuration: optional centre circle plus rings.

    Each ring is (rho, theta0) and contributes k circles at angles
    theta0 + 2*pi*j/k.
    """

    k: int
    rings: tuple[tuple[float, float], ...]
    center: bool = True

    @property
    def n(self) -> int:
        return len(self.rings) * self.k + (1 if self.center else 0)

    def centers(self) -> np.ndarray:
        out = [np.zeros((1, 2))] if self.center else []
        for rho, th in self.rings:
            a = th + 2 * np.pi * np.arange(self.k) / self.k
            out.append(np.stack([rho * np.cos(a), rho * np.sin(a)], axis=1))
        return np.concatenate(out, axis=0)
