"""Rigorous global optimisation inside rotationally symmetric classes.

CLASSES.  If 25 unit disks are invariant under rotation by 2 pi / k about the
origin then every orbit has size k, except for at most one orbit sitting at the
origin (a non-zero point has trivial stabiliser under a rotation of order k).
Hence 25 = k m + [centre], which forces

    k = 25 : one orbit of 25                (params: 1 radius)
    k = 24 : centre + one orbit of 24       (1 radius)
    k = 12 : centre + two orbits of 12      (2 radii + 1 angle)
    k =  8 : centre + three orbits of 8     (3 radii + 2 angles)   <-- the record
    k =  6 : centre + four orbits of 6      (4 radii + 3 angles)
    k =  5 : five orbits of 5               (5 radii + 4 angles)

One angle is set to 0 by a global rotation (choose the ring of smallest radius),
and the radii may then be assumed non-decreasing.

BOUND.  Let H(C,R) = max_{|p| <= R} min_i |p - c_i|, computed exactly by
hole.deepest_hole.  Then disk(0,R) is covered iff H(C,R) <= 1, and H is
1-Lipschitz in the perturbation size max_i |c_i - c_i'|:

    |H(C,R) - H(C',R)| <= max_i |c_i - c_i'| .

For a parameter box V with centre v0, every c_i(v), v in V, satisfies
|c_i(v) - c_i(v0)| <= max_j (h_j^rho + rho_max h_j^theta) =: delta(V),
where h are the half-widths.  Therefore

    H(C(v0), R) - delta(V) > 1     ==>   NO v in V gives a covering of disk(0,R).

That test is sharp (no interval over-estimation), so the branch and bound
terminates with a number of nodes proportional to 2^dim times the number of
bisection levels needed near the optimum.

SIDE LEMMA (rings pushed to infinity).  The boxes only cover radii in
[0, R+1].  If some ring had radius > R+1 its k disks would miss the open disk of
radius R entirely, leaving 25 - k <= 20 disks to cover it; the certified
fractional bound phi(4.1816) = 22.88 > 20 rules that out for every k >= 5.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from hole import deepest_hole, deepest_hole_symmetric

R25 = math.sqrt(6) + math.sqrt(3)


@dataclass
class Klass:
    k: int
    m: int
    center: bool
    fixed_angles: tuple[float, ...] | None = None

    @property
    def n_ang(self) -> int:
        return 0 if self.fixed_angles is not None else max(self.m - 1, 0)

    @property
    def dim(self) -> int:
        return self.m + self.n_ang

    def n(self) -> int:
        return self.k * self.m + (1 if self.center else 0)

    def _angle(self, j: int, v: np.ndarray) -> float:
        if self.fixed_angles is not None:
            return self.fixed_angles[j]
        return 0.0 if j == 0 else float(v[self.m + j - 1])

    def centers(self, v: np.ndarray) -> np.ndarray:
        pts = [np.zeros((1, 2))] if self.center else []
        for j in range(self.m):
            rho = v[j]
            th = self._angle(j, v)
            a = th + np.arange(self.k) * 2 * math.pi / self.k
            pts.append(np.stack([rho * np.cos(a), rho * np.sin(a)], axis=1))
        return np.concatenate(pts, axis=0)

    def initial_box(self, R: float) -> np.ndarray:
        b = np.zeros((self.dim, 2))
        for j in range(self.m):
            b[j] = (0.0, R + 1.0)
        for j in range(self.n_ang):
            b[self.m + j] = (0.0, 2 * math.pi / self.k)
        return b

    def delta_terms(self, box: np.ndarray) -> np.ndarray:
        """Per-ring bound on |c_i(v) - c_i(v0)| for v in the box.

        |rho e^{i a} - rho_0 e^{i a_0}| <= |rho - rho_0| + rho_0 |a - a_0|, and
        rho_0 <= box[j,1], so ring j moves by at most h_rho + box[j,1] * h_theta.
        """
        h = 0.5 * (box[:, 1] - box[:, 0])
        out = np.empty(self.m)
        for j in range(self.m):
            if self.fixed_angles is not None or j == 0:
                ang = 0.0
            else:
                ang = h[self.m + j - 1]
            out[j] = h[j] + box[j, 1] * ang
        return out

    def delta(self, box: np.ndarray, rmax: float = 0.0) -> float:
        return float(self.delta_terms(box).max())

    def split_dim(self, box: np.ndarray) -> int:
        """Split the coordinate that contributes most to delta."""
        h = 0.5 * (box[:, 1] - box[:, 0])
        contrib = np.zeros(self.dim)
        j_star = int(np.argmax(self.delta_terms(box)))
        contrib[j_star] = h[j_star]
        if self.fixed_angles is None and j_star > 0:
            contrib[self.m + j_star - 1] = box[j_star, 1] * h[self.m + j_star - 1]
        if contrib.max() <= 0:
            return int(np.argmax(h))
        return int(np.argmax(contrib))


@dataclass(order=True)
class Node:
    key: float
    box: np.ndarray = field(compare=False)


def _alpha_max_interval(t: float, lo: float, hi: float) -> float:
    """max of alpha(t, rho) over rho in [lo, hi]  (0 if the arc is always empty)."""
    if t <= 0.0:
        return 0.0
    if t <= 1.0 and lo <= 1.0 - t:
        return math.pi
    if t > 1.0:
        star = math.sqrt(t * t - 1.0)
        if lo <= star <= hi:
            return math.asin(1.0 / t)
    best = 0.0
    for rho in (lo, hi):
        if rho <= 1e-15:
            best = max(best, math.pi if t <= 1.0 else 0.0)
            continue
        if abs(t - rho) > 1.0:
            continue
        if t + rho <= 1.0:
            return math.pi
        c = (t * t + rho * rho - 1.0) / (2.0 * t * rho)
        best = max(best, math.acos(min(max(c, -1.0), 1.0)))
    return best


def radial_arc_eliminates(kl: Klass, box: np.ndarray, R: float,
                          ts: np.ndarray) -> bool:
    """Cheap test: for some t <= R, even the most generous radii in the box give
    total arc length < 2 pi on S_t, so S_t cannot be covered.  Ignores angles."""
    for t in ts:
        tot = math.pi if (kl.center and t <= 1.0) else 0.0
        for j in range(kl.m):
            tot += kl.k * _alpha_max_interval(t, box[j, 0], box[j, 1])
            if tot >= math.pi:
                break
        if tot < math.pi - 1e-12:
            return True
    return False


def prove_class(kl: Klass, R: float, max_nodes: int = 2_000_000,
                min_width: float = 1e-12, verbose: bool = False,
                n_arc: int = 60, safety: float = 1e-11):
    """safety absorbs double-precision error: a box is only eliminated when
    H(C(v0),R) - delta(V) > 1 + safety."""
    rmax = R + 1.0
    box0 = kl.initial_box(R)
    heap: list[Node] = []
    ts = np.linspace(R / n_arc, R, n_arc)
    stats = {"arc": 0, "lip": 0}

    def test(b: np.ndarray):
        """Returns the elimination margin (>1 means the box is eliminated)."""
        if radial_arc_eliminates(kl, b, R, ts):
            stats["arc"] += 1
            return np.inf
        v0 = b.mean(axis=1)
        H = deepest_hole_symmetric(kl.centers(v0), R, kl.k)
        return H - kl.delta(b)

    if test(box0) > 1.0 + safety:
        return {"proved": True, "nodes": 1, "n_survivors": 0, "stats": stats}
    heapq.heappush(heap, Node(test(box0), box0))  # not eliminated
    nodes = 0
    survivors: list[np.ndarray] = []
    while heap:
        node = heapq.heappop(heap)
        nodes += 1
        if nodes > max_nodes:
            survivors.append(node.box)
            survivors += [nd.box for nd in heap]
            break
        b = node.box
        w = b[:, 1] - b[:, 0]
        if w.max() < min_width:
            survivors.append(b)
            continue
        j = kl.split_dim(b)
        mid = 0.5 * (b[j, 0] + b[j, 1])
        for lo, hi in ((b[j, 0], mid), (mid, b[j, 1])):
            nb = b.copy()
            nb[j] = (lo, hi)
            # WLOG the ring radii are non-decreasing
            if any(nb[t, 0] > nb[t + 1, 1] for t in range(kl.m - 1)):
                continue
            m2 = test(nb)
            if m2 <= 1.0 + safety:
                heapq.heappush(heap, Node(m2, nb))
            else:
                stats["lip"] += 1
        if verbose and nodes % 50_000 == 0:
            print(f"      nodes={nodes:8d} open={len(heap):7d} "
                  f"best_margin={heap[0].key if heap else float('nan'):.6f}")
    return {"proved": len(survivors) == 0, "nodes": nodes,
            "n_survivors": len(survivors), "stats": stats,
            "survivors": [s.copy() for s in survivors[:5]]}


def main() -> None:
    import sys

    print("=" * 78)
    print("Rigorous branch and bound inside rotationally symmetric classes")
    print("Criterion: H(C(v0),R) - delta(box) > 1  eliminates the whole box.")
    print("=" * 78)

    targets = [float(x) for x in sys.argv[1:]] or [4.25, 4.20, 4.19, 4.1816]
    classes = [Klass(25, 1, False), Klass(24, 1, True), Klass(12, 2, True),
               Klass(8, 3, True)]
    for R in targets:
        print(f"\nR_target = {R:.6f}   (sqrt6+sqrt3 = {R25:.6f}, "
              f"excess = {R - R25:.2e})")
        for kl in classes:
            res = prove_class(kl, R, max_nodes=600_000, verbose=False)
            if res["proved"]:
                tag = "PROVED: no configuration in this class covers disk(0,R)"
            else:
                tag = f"inconclusive, {res['n_survivors']} boxes left"
            print(f"  k={kl.k:2d} dim={kl.dim}: nodes={res['nodes']:8d}  {tag}")


if __name__ == "__main__":
    main()
