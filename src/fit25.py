"""Pin down the n=25 configuration exactly.

The picture on Friedman's page has 8-fold dihedral symmetry:
    1 centre circle
  + ring A: 8 circles at (rho_A, 22.5 deg + 45 deg * j)
  + ring B: 8 circles at (rho_B,  0.0 deg + 45 deg * j)
  + ring C: 8 circles at (rho_C, 22.5 deg + 45 deg * j)

We first optimise inside that symmetry class, then confirm the numbers against
closed forms.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from geom import RingConfig, covering_radius

R_TARGET = math.sqrt(6) + math.sqrt(3)


def build(v: np.ndarray) -> np.ndarray:
    """v = [rhoA, thA, rhoB, thB, rhoC, thC]"""
    return RingConfig(
        k=8,
        rings=((v[0], v[1]), (v[2], v[3]), (v[4], v[5])),
        center=True,
    ).centers()


def neg_R(v: np.ndarray) -> float:
    return -covering_radius(build(v))


def local_maximise(v0: np.ndarray, iters: int = 6) -> tuple[np.ndarray, float]:
    v = v0.copy()
    best = -neg_R(v)
    step = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    for _ in range(iters):
        res = minimize(neg_R, v, method="Nelder-Mead",
                       options={"xatol": 1e-13, "fatol": 1e-14, "maxfev": 40000,
                                "initial_simplex": None})
        if -res.fun > best:
            best, v = -res.fun, res.x
        step *= 0.3
    return v, best


def main() -> None:
    px = np.load("data/centers_25.npy")
    print(f"picture-derived covering radius: {covering_radius(px):.9f}")
    print(f"target sqrt(6)+sqrt(3)        : {R_TARGET:.9f}")

    rho = np.hypot(px[:, 0], px[:, 1])
    print("\nrho values from picture:", np.round(np.sort(rho), 4))

    v0 = np.array([1.8478, math.pi / 8, 3.409, 0.0, 3.691, math.pi / 8])
    print(f"\nstart R = {covering_radius(build(v0)):.10f}")
    v, best = local_maximise(v0)
    print(f"optimised R = {best:.12f}   (target {R_TARGET:.12f})")
    print(f"  gap to target = {best - R_TARGET:.3e}")
    names = ["rhoA", "thA", "rhoB", "thB", "rhoC", "thC"]
    for nm, val in zip(names, v):
        extra = f" = {math.degrees(val):.6f} deg" if nm.startswith("th") else ""
        print(f"  {nm} = {val:.12f}{extra}")
    np.save("data/opt25_params.npy", v)
    np.save("data/opt25_centers.npy", build(v))


if __name__ == "__main__":
    main()
