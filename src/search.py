"""Unrestricted numerical search for coverings of disk(0,R) by 25 unit disks.

For a fixed R the question "can 25 unit disks cover disk(0,R)?" is the continuous
25-centre problem: minimise over C the value

    F(C) = max_{p in disk(0,R)} min_i |p - c_i|

and ask whether F(C) <= 1.  We minimise F by the standard alternation
(Lloyd/Chebyshev iteration for k-centre):

    1. discretise disk(0,R) by a fine net;
    2. assign every net point to its nearest centre;
    3. replace each centre by the centre of the minimum enclosing circle of its
       cluster (that is the exact 1-cluster optimum);
    4. repeat.

Step 3 never increases F on the net, so the iteration converges.  Many random
restarts plus a final continuous polish give a good local search.  The exact
covering radius of every candidate is then evaluated with geom.covering_radius,
which does not use the net at all.
"""

from __future__ import annotations

import math

import numpy as np

from config25 import R25, centers25
from geom import covering_radius


def polar_net(R: float, n_r: int = 90) -> np.ndarray:
    """Near-uniform net of disk(0,R)."""
    pts = [np.zeros((1, 2))]
    for k in range(1, n_r + 1):
        r = R * k / n_r
        m = max(6, int(round(2 * math.pi * r / (R / n_r))))
        a = np.linspace(0, 2 * math.pi, m, endpoint=False)
        pts.append(np.stack([r * np.cos(a), r * np.sin(a)], axis=1))
    return np.concatenate(pts, axis=0)


def mec(pts: np.ndarray, iters: int = 300) -> tuple[np.ndarray, float]:
    """Minimum enclosing circle by Chebyshev-style shrinking (Badoiu-Clarkson)."""
    c = pts.mean(axis=0)
    for k in range(1, iters + 1):
        d = np.linalg.norm(pts - c, axis=1)
        f = pts[int(np.argmax(d))]
        c = c + (f - c) / (k + 1)
    return c, float(np.linalg.norm(pts - c, axis=1).max())


def kcenter_alternate(net: np.ndarray, C: np.ndarray, iters: int = 200):
    n = len(C)
    best = np.inf
    for _ in range(iters):
        d = np.linalg.norm(net[:, None, :] - C[None, :, :], axis=2)
        lab = np.argmin(d, axis=1)
        val = float(d[np.arange(len(net)), lab].max())
        newC = C.copy()
        for i in range(n):
            cl = net[lab == i]
            if len(cl) == 0:
                # reseed an empty cluster at the currently worst-covered point
                worst = int(np.argmax(d.min(axis=1)))
                newC[i] = net[worst]
                continue
            newC[i] = mec(cl)[0]
        if np.allclose(newC, C, atol=1e-13):
            C = newC
            break
        C = newC
        best = min(best, val)
    d = np.linalg.norm(net[:, None, :] - C[None, :, :], axis=2)
    return C, float(d.min(axis=1).max())


def search(R: float, restarts: int = 40, n_r: int = 80, seed: int = 0,
           seed_config: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    net = polar_net(R, n_r)
    best = (np.inf, None)
    starts = []
    if seed_config is not None:
        starts.append(seed_config.copy())
    for _ in range(restarts):
        # random start: mix of uniform and ring-structured
        if rng.random() < 0.5:
            a = rng.uniform(0, 2 * math.pi, 25)
            rr = R * np.sqrt(rng.uniform(0, 1, 25))
            starts.append(np.stack([rr * np.cos(a), rr * np.sin(a)], axis=1))
        else:
            k = int(rng.integers(3, 9))
            rings = []
            left = 25
            if rng.random() < 0.6:
                rings.append(np.zeros((1, 2)))
                left -= 1
            while left >= k:
                rho = rng.uniform(0.3, R)
                ph = rng.uniform(0, 2 * math.pi)
                a = ph + np.arange(k) * 2 * math.pi / k
                rings.append(np.stack([rho * np.cos(a), rho * np.sin(a)], axis=1))
                left -= k
            if left:
                a = rng.uniform(0, 2 * math.pi, left)
                rr = R * np.sqrt(rng.uniform(0, 1, left))
                rings.append(np.stack([rr * np.cos(a), rr * np.sin(a)], axis=1))
            starts.append(np.concatenate(rings, axis=0)[:25])
    for C0 in starts:
        C, val = kcenter_alternate(net, C0.astype(float))
        if val < best[0]:
            best = (val, C)
    return best


def main() -> None:
    print("=" * 78)
    print("Unrestricted search: can 25 unit disks cover more than sqrt6+sqrt3?")
    print("=" * 78)
    C0 = centers25()
    print(f"reference configuration: exact covering radius = "
          f"{covering_radius(C0):.12f}   (sqrt6+sqrt3 = {R25:.12f})")

    print("\nFor each R we minimise the net k-centre value; a value <= 1 would")
    print("mean 25 unit disks cover disk(0,R).  (Net values slightly UNDERstate")
    print("the true requirement, so val <= 1 is only a hint, checked exactly after.)")
    print(f"\n{'R':>10} {'best net value':>15} {'exact R of best':>17}  {'beats R25?':>11}")
    for R in (R25, 4.19, 4.20, 4.22, 4.25, 4.30):
        val, C = search(R, restarts=30, n_r=70, seed=1,
                        seed_config=C0 * (R / R25))
        exact = covering_radius(C)
        print(f"{R:10.6f} {val:15.9f} {exact:17.9f}  "
              f"{'YES' if exact > R25 + 1e-9 else 'no':>11}")

    print("\nlarger search at R = sqrt6+sqrt3 (does anything match or beat it?)")
    val, C = search(R25, restarts=120, n_r=90, seed=7)
    print(f"  best net value = {val:.9f}, exact covering radius = "
          f"{covering_radius(C):.9f}")
    np.save("data/search_best.npy", C)


if __name__ == "__main__":
    main()
