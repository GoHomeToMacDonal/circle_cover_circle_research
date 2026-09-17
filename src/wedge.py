"""Reduce the covering of disk(R25) to three one-variable inequalities.

The configuration has dihedral symmetry D_8 (order 16): rotation by pi/4 plus
reflection in the line theta = 0.  A fundamental domain is the wedge

    W = { t e^{i theta} : 0 <= t <= R,  0 <= theta <= pi/8 }.

On the circle S_t a unit disk centred at (rho, phi) covers the closed arc
[phi - alpha, phi + alpha] with

    cos alpha(t; rho) = (t^2 + rho^2 - 1) / (2 t rho),

empty when |t - rho| > 1.  Only four disks matter inside W:

    O  at rho = 0            (covers S_t entirely for t <= 1)
    A0 at (rho_A, pi/8)      arc centred on the right edge of W
    B0 at (rho_B, 0)         arc centred on the left edge of W
    C0 at (rho_C, pi/8)      arc centred on the right edge of W

so W is covered iff for every t the arcs reach across the angular width pi/8:

    (I)   1        <= t <= 1+sqrt2 :  alpha_A(t)               >= pi/8
    (II)  1+sqrt2  <= t <= t_x     :  alpha_A(t) + alpha_B(t)  >= pi/8
    (III) t_x      <= t <= R       :  alpha_C(t) + alpha_B(t)  >= pi/8

with t_x = sqrt(5 + 2 sqrt2) the radius where alpha_A = alpha_C (because
rho_C = 2 rho_A).
"""

from __future__ import annotations

import math

import numpy as np

U = math.cos(math.pi / 8)
RHO_A, RHO_B, RHO_C = 2 * U, 2 + math.sqrt(2), 4 * U
R25 = math.sqrt(6) + math.sqrt(3)
T1 = 1.0
T2 = 1 + math.sqrt(2)
TX = math.sqrt(5 + 2 * math.sqrt(2))
W = math.pi / 8


def alpha(t: float, rho: float) -> float:
    """Half-angle of the arc cut on S_t by a unit disk at distance rho."""
    if t <= 0 or abs(t - rho) > 1:
        return float("nan")
    c = (t * t + rho * rho - 1) / (2 * t * rho)
    return math.acos(min(max(c, -1.0), 1.0))


def main() -> None:
    print(f"rho_A = {RHO_A:.15f}   rho_A - 1 = {RHO_A-1:.15f}")
    print(f"rho_B = {RHO_B:.15f}   rho_B - 1 = {RHO_B-1:.15f}  (= 1+sqrt2)")
    print(f"rho_C = {RHO_C:.15f}   rho_C - 1 = {RHO_C-1:.15f}")
    print(f"t_x   = {TX:.15f}      R = {R25:.15f}")
    print(f"check rho_C == 2 rho_A : {abs(RHO_C - 2*RHO_A):.2e}")
    print(f"check rho_B == rho_A^2 : {abs(RHO_B - RHO_A**2):.2e}")
    print(f"check t_x^2 == 2 rho_A^2 + 1 : {abs(TX**2 - (2*RHO_A**2+1)):.2e}")

    print("\n(I)  alpha_A - pi/8 on [1, 1+sqrt2]  (expect 0 at both ends, >0 inside)")
    for t in np.linspace(T1, T2, 11):
        print(f"   t={t:.9f}  slack={math.degrees(alpha(t,RHO_A)-W):+10.6f} deg")

    print("\n(II) alpha_A + alpha_B - pi/8 on [1+sqrt2, t_x]")
    for t in np.linspace(T2, TX, 13):
        s = alpha(t, RHO_A) + alpha(t, RHO_B) - W
        print(f"   t={t:.9f}  slack={math.degrees(s):+10.6f} deg")

    print("\n(III) alpha_C + alpha_B - pi/8 on [t_x, R]")
    for t in np.linspace(TX, R25, 13):
        s = alpha(t, RHO_C) + alpha(t, RHO_B) - W
        print(f"   t={t:.9f}  slack={math.degrees(s):+10.6f} deg")

    print("\nminimum slack of (II) and (III) on fine grids:")
    g2 = np.linspace(T2, TX, 200001)
    s2 = np.array([alpha(t, RHO_A) + alpha(t, RHO_B) - W for t in g2])
    g3 = np.linspace(TX, R25, 200001)
    s3 = np.array([alpha(t, RHO_C) + alpha(t, RHO_B) - W for t in g3])
    print(f"  (II)  min = {math.degrees(np.nanmin(s2)):+.9f} deg at t={g2[np.nanargmin(s2)]:.6f}")
    print(f"  (III) min = {math.degrees(np.nanmin(s3)):+.9f} deg at t={g3[np.nanargmin(s3)]:.6f}")
    # interior minima excluding endpoints
    in2 = s2[100:-100]
    in3 = s3[100:-100]
    print(f"  (II)  interior min = {math.degrees(in2.min()):+.9f} deg")
    print(f"  (III) interior min = {math.degrees(in3.min()):+.9f} deg")
    print(f"\n  alpha_B(R) = {math.degrees(alpha(R25,RHO_B)):.9f} deg")
    print(f"  alpha_C(R) = {math.degrees(alpha(R25,RHO_C)):.9f} deg")
    print(f"  sum        = {math.degrees(alpha(R25,RHO_B)+alpha(R25,RHO_C)):.9f} deg (= 22.5)")


if __name__ == "__main__":
    main()
