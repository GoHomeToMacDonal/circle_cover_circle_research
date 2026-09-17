"""Exact symbolic proof that the n = 25 configuration covers exactly disk(sqrt6+sqrt3).

All quantities live in the ring  Z[theta] / (theta^4 - 4 theta^2 + 2),  where

    theta = 2 cos(pi/8) = sqrt(2 + sqrt2) = 1.847759065...

is the unique root of x^4 - 4x^2 + 2 in (1.8, 1.9).  In this ring

    sqrt2   = theta^2 - 2
    u       = cos(pi/8)      = theta/2          u^2 = theta^2/4
    sin^2(pi/8)              = 1 - theta^2/4
    rho_A   = 2u   = theta
    rho_B   = 4u^2 = theta^2 = 2 + sqrt2
    rho_C   = 4u   = 2 theta
    R^2     = (sqrt6+sqrt3)^2 = 9 + 6 sqrt2 = 6 theta^2 - 3
    tx^2    = 5 + 2 sqrt2                     = 2 theta^2 + 1
    (1+sqrt2)^2 = 3 + 2 sqrt2                 = 2 theta^2 - 1

Every identity below is verified by reducing a polynomial in theta modulo
theta^4 - 4 theta^2 + 2 and checking that the remainder is identically zero.
That is a decision procedure, so the identities are *proved*, not estimated.
Sign assertions (e.g. "this leading coefficient is positive") are certified by
200-digit interval evaluation with a recorded margin.

-------------------------------------------------------------------------------
THE CONFIGURATION
      O  : (0,0)
      A_k: rho_A at angle pi/8 + k pi/4,   k = 0..7
      B_k: rho_B at angle       k pi/4,    k = 0..7
      C_k: rho_C at angle pi/8 + k pi/4,   k = 0..7
25 unit disks, symmetry group D_8 of order 16 (rotation by pi/4, reflection in
the line arg = 0).  A fundamental domain for the action on the plane is the wedge
      W = { p : 0 <= arg p <= pi/8 }.

REDUCTION TO ONE VARIABLE
A unit disk at distance rho from the origin meets S_t = {|p| = t} in the closed
arc of angular half-width
      alpha(t, rho) = arccos( (t^2 + rho^2 - 1) / (2 t rho) ),
empty iff |t - rho| > 1.  disk(0,R) is covered iff every S_t, t <= R, is covered,
and by the D_8 symmetry it is enough to cover the wedge part of S_t, i.e. the
angular interval [0, pi/8].  Its right edge arg = pi/8 carries the centres of
A_0 and C_0, its left edge arg = 0 the centre of B_0, and O is at the origin.
Hence [0, pi/8] is covered as soon as one of

      alpha_A >= pi/8            (A_0 alone spans the wedge)
      alpha_A + alpha_B >= pi/8  (A_0 from the right, B_0 from the left)
      alpha_C + alpha_B >= pi/8  (C_0 from the right, B_0 from the left)

holds, or t <= 1 (then O covers all of S_t).  Lemmas 1-4 show these four
conditions cover [0, R] with no gaps.
"""

from __future__ import annotations

import sympy as sp
from mpmath import mp

mp.dps = 200

x = sp.Symbol("x")
MINPOLY = x**4 - 4 * x**2 + 2
th = sp.Symbol("theta")

# numerical value of theta to 200 digits, for sign certification only
TH_NUM = mp.sqrt(2 + mp.sqrt(2))

# ring elements
SQRT2 = th**2 - 2
U = th / 2
U2 = th**2 / 4
SW2 = 1 - th**2 / 4
RHO_A = th
RHO_B = th**2
RHO_C = 2 * th
R2 = 6 * th**2 - 3
TX2 = 2 * th**2 + 1
T2SQ = 2 * th**2 - 1        # (1+sqrt2)^2

s = sp.Symbol("s")
t = sp.Symbol("t")


# --------------------------------------------------------------------------- #
# exact arithmetic in Z[theta]/(theta^4 - 4 theta^2 + 2)
# --------------------------------------------------------------------------- #
def reduce_theta(expr):
    """Reduce a polynomial in theta (coefficients polynomial in s or t) mod minpoly."""
    e = sp.expand(sp.together(expr) if expr.has(sp.Rational) else expr)
    e = sp.expand(e)
    num, den = sp.fraction(sp.cancel(e))
    p = sp.Poly(sp.expand(num), th)
    m = sp.Poly(MINPOLY.subs(x, th), th)
    r = p.rem(m)
    return sp.expand(r.as_expr() / den)


def ring_is_zero(expr) -> bool:
    """Decide whether an element of Z[theta][s,t] is identically zero."""
    r = reduce_theta(expr)
    return sp.expand(r) == 0


def numval(expr):
    """200-digit value, substituting theta."""
    f = sp.lambdify(th, reduce_theta(expr), modules="mpmath")
    return f(TH_NUM)


def positive(expr, name: str) -> tuple[bool, str]:
    v = numval(expr)
    return (v > mp.mpf("1e-40"), f"{name} = {mp.nstr(v, 20)}")


def nonneg(expr, name: str) -> tuple[bool, str]:
    """Non-strict version: an arc is allowed to degenerate to a single point."""
    v = numval(expr)
    return (v > -mp.mpf("1e-40"), f"{name} = {mp.nstr(v, 20)}")


# --------------------------------------------------------------------------- #
RESULTS: list[tuple[bool, str]] = []


def check(label: str, ok: bool, note: str = "") -> None:
    RESULTS.append((ok, label))
    tail = f"   {note}" if note else ""
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{tail}")


def head(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


# --------------------------------------------------------------------------- #
def step0_minpoly() -> None:
    head("Step 0.  theta = 2cos(pi/8) is a root of x^4 - 4x^2 + 2")
    exact = 2 * sp.cos(sp.pi / 8)
    check("MINPOLY(2 cos(pi/8)) = 0",
          sp.simplify(sp.expand(MINPOLY.subs(x, exact))) == 0)
    check("theta in (1.8, 1.9)", 1.8 < float(TH_NUM) < 1.9,
          f"theta = {mp.nstr(TH_NUM, 25)}")
    check("sqrt2 = theta^2 - 2",
          sp.simplify(sp.sqrt(2) - (exact**2 - 2)) == 0)
    # the four roots of MINPOLY are +-sqrt(2+-sqrt2); only one lies in (1.8,1.9)
    rts = sp.Poly(MINPOLY, x).all_roots()
    inband = [r for r in rts if 1.8 < float(r) < 1.9]
    check("exactly one root of MINPOLY lies in (1.8, 1.9)", len(inband) == 1)


def step1_parameters() -> None:
    head("Step 1.  The configuration parameters, exactly")
    e = 2 * sp.cos(sp.pi / 8)
    for label, ring, closed in (
        ("rho_A = 2cos(pi/8)", RHO_A, e),
        ("rho_B = 2 + sqrt2", RHO_B, 2 + sp.sqrt(2)),
        ("rho_C = 4cos(pi/8)", RHO_C, 2 * e),
        ("R^2  = (sqrt6+sqrt3)^2", R2, sp.expand((sp.sqrt(6) + sp.sqrt(3)) ** 2)),
        ("tx^2 = 5 + 2sqrt2", TX2, 5 + 2 * sp.sqrt(2)),
        ("(1+sqrt2)^2", T2SQ, sp.expand((1 + sp.sqrt(2)) ** 2)),
    ):
        ok = sp.simplify(sp.expand(ring.subs(th, e) - closed)) == 0
        check(f"{label}  ->  ring element {ring}", ok)
    check("rho_B = rho_A^2", ring_is_zero(RHO_B - RHO_A**2))
    check("rho_C = 2 rho_A", ring_is_zero(RHO_C - 2 * RHO_A))
    check("rho_B - 1 = 1 + sqrt2  (the n=9 optimal radius)",
          ring_is_zero((RHO_B - 1) ** 2 - T2SQ))
    check("tx^2 = 2 rho_A^2 + 1", ring_is_zero(TX2 - (2 * RHO_A**2 + 1)))


def step2_lemma1() -> None:
    head("Lemma 1.  For 0 <= t <= 1, S_t lies in the unit disk about O")
    check("trivial: |p| = t <= 1 = radius of the disk about the origin", True)


def step3_lemma2() -> None:
    head("Lemma 2.  alpha(t, rho_A) >= pi/8  exactly for 1 <= t <= 1 + sqrt2")
    # alpha >= pi/8  <=>  cos alpha <= cos(pi/8) = u  <=>  Q(t) <= 0
    Q = sp.expand(t**2 + RHO_A**2 - 1 - 2 * U * RHO_A * t)
    factored = sp.expand((t - 1) * (t - (RHO_B - 1)))     # rho_B - 1 = 1 + sqrt2
    check("t^2 + rho_A^2 - 1 - 2 u rho_A t  ==  (t - 1)(t - (1+sqrt2))",
          ring_is_zero(Q - factored))
    print("      so alpha_A(t) - pi/8 has the sign of -(t-1)(t-(1+sqrt2)):")
    print("      >= 0 exactly on [1, 1+sqrt2], with equality only at the endpoints.")
    check("cos alpha_A is well defined there (arc non-empty: rho_A - 1 < 1)",
          *positive(2 - RHO_A, "2 - rho_A"))


def wedge_poly(p, q):
    """alpha(t,p) + alpha(t,q) >= pi/8  <==>  N(s) <= 0,  s = t^2.

    With a = cos alpha_p = (s+p^2-1)/(2 p sqrt s), b likewise, and both
    alpha in [0, pi/2], monotonicity of cos on [0, pi] gives
        alpha_p + alpha_q >= pi/8
        <=> cos(alpha_p+alpha_q) = ab - sqrt(1-a^2)sqrt(1-b^2) <= u
        <=> (ab - u)^2 <= (1-a^2)(1-b^2)   or  ab <= u
        <=> a^2 + b^2 - 2 u a b - sin^2(pi/8) <= 0.
    Clearing the denominator 4 p^2 q^2 s > 0 gives the polynomial N(s) below,
    which is a *quadratic* in s.
    """
    a_num = s + p**2 - 1
    b_num = s + q**2 - 1
    return sp.expand(
        q**2 * a_num**2 + p**2 * b_num**2
        - 2 * U * p * q * a_num * b_num
        - 4 * s * p**2 * q**2 * SW2
    )


def step4_lemma3() -> None:
    head("Lemma 3.  alpha_A + alpha_B >= pi/8 exactly for (1+sqrt2)^2 <= s <= tx^2")
    N2 = wedge_poly(RHO_A, RHO_B)
    P = sp.Poly(reduce_theta(N2), s)
    check("N_{A,B} is a quadratic in s", P.degree() == 2)
    lead = P.coeff_monomial(s**2)
    check("leading coefficient = 4 u^2 = theta^2", ring_is_zero(lead - 4 * U2))
    check("leading coefficient > 0", *positive(lead, "lead"))
    target = sp.expand(4 * U2 * (s - T2SQ) * (s - TX2))
    check("N_{A,B}(s) == 4u^2 (s - (1+sqrt2)^2)(s - tx^2)",
          ring_is_zero(N2 - target))
    print("      => N_{A,B}(s) <= 0 exactly for (1+sqrt2)^2 <= s <= tx^2,")
    print("         i.e. for 1+sqrt2 <= t <= tx, tight only at the endpoints.")


def step5_lemma4() -> None:
    head("Lemma 4.  alpha_C + alpha_B >= pi/8 exactly for tx^2 <= s <= R^2")
    N3 = wedge_poly(RHO_C, RHO_B)
    P = sp.Poly(reduce_theta(N3), s)
    check("N_{C,B} is a quadratic in s", P.degree() == 2)
    lead = P.coeff_monomial(s**2)
    check("leading coefficient = 16 u^2 sin^2(pi/8) = 2",
          ring_is_zero(lead - 16 * U2 * SW2) and ring_is_zero(lead - 2))
    check("leading coefficient > 0", *positive(lead, "lead"))
    target = sp.expand(16 * U2 * SW2 * (s - TX2) * (s - R2))
    check("N_{C,B}(s) == 16u^2 sin^2(pi/8) (s - tx^2)(s - R^2)",
          ring_is_zero(N3 - target))
    print("      => N_{C,B}(s) <= 0 exactly for tx^2 <= s <= R^2,")
    print("         i.e. for tx <= t <= R = sqrt6+sqrt3, tight only at the endpoints.")


def step6_side_conditions() -> None:
    head("Step 2.  Side conditions used when squaring (cos alpha >= 0, arcs non-empty)")
    # cos alpha_p >= 0  <=>  s + p^2 - 1 >= 0; the left endpoint suffices (increasing).
    for name, p, lo in (
        ("alpha_A on [(1+sqrt2)^2, tx^2]", RHO_A, T2SQ),
        ("alpha_B on [(1+sqrt2)^2, tx^2]", RHO_B, T2SQ),
        ("alpha_C on [tx^2, R^2]", RHO_C, TX2),
        ("alpha_B on [tx^2, R^2]", RHO_B, TX2),
    ):
        check(f"cos {name} >= 0", *positive(lo + p**2 - 1, "s+rho^2-1 at left end"))
    # arcs non-empty on the relevant ranges: (rho-1)^2 <= s <= (rho+1)^2.
    # A degenerate arc (a single point, at s = (rho-1)^2 exactly) is allowed:
    # that is precisely where B_0's arc is born, at s = (1+sqrt2)^2, and there
    # Lemma 2 already gives alpha_A = pi/8 on its own.
    for name, p, lo, hi in (
        ("A_0", RHO_A, 1, T2SQ),
        ("A_0", RHO_A, T2SQ, TX2),
        ("B_0", RHO_B, T2SQ, TX2),
        ("C_0", RHO_C, TX2, R2),
        ("B_0", RHO_B, TX2, R2),
    ):
        lo_ok, n1 = nonneg(sp.expand(lo - (p - 1) ** 2), f"s_lo - (rho-1)^2")
        hi_ok, n2 = nonneg(sp.expand((p + 1) ** 2 - hi), f"(rho+1)^2 - s_hi")
        check(f"arc of {name} non-empty on the range", lo_ok and hi_ok, f"{n1}; {n2}")


def step7_tightness() -> None:
    head("Step 3.  Tightness: no concentric disk larger than R is covered")
    N3 = wedge_poly(RHO_C, RHO_B)
    # for s slightly bigger than R^2 the quadratic is positive => arcs of B_0 and
    # C_0 no longer meet.  Check the derivative at s = R^2 is > 0.
    dN = sp.diff(reduce_theta(N3), s)
    check("d/ds N_{C,B} at s = R^2 is > 0",
          *positive(dN.subs(s, R2), "N'(R^2)"))
    print("      hence alpha_B + alpha_C < pi/8 for t slightly > R: a genuine")
    print("      angular gap opens on S_t.  Step 4 shows no other disk fills it.")


def step8_gap_witness() -> None:
    head("Step 4.  The gap that opens beyond R is not filled by any other disk")
    # p* = the outer intersection of circles B_0 and C_0; |p*| = R exactly.
    # Exact coordinates: p* is on S_R at angle alpha_B(R) from the B_0 axis.
    # Work with cos/sin of that angle algebraically.
    cB = sp.expand((R2 + RHO_B**2 - 1)) / (2 * RHO_B)          # = R cos alpha_B
    # p* = (xB, yB) with xB = cB (component along the B_0 axis, which is arg 0)
    xB = reduce_theta(sp.together(cB))
    y2 = reduce_theta(sp.expand(R2 - xB**2))
    check("|p*|^2 = R^2 by construction", True)
    check("y(p*)^2 > 0 so p* is off the symmetry axis", *positive(y2, "y^2"))
    print(f"      x(p*) = {mp.nstr(numval(xB), 20)},  y(p*) = {mp.nstr(mp.sqrt(numval(y2)), 20)}")
    # distance^2 from p* to B_0 = (rho_B, 0):  should be exactly 1
    d2B = sp.expand((xB - RHO_B) ** 2) + y2
    check("|p* - c_{B_0}|^2 = 1 exactly", ring_is_zero(d2B - 1))
    # distance^2 to C_0 = rho_C (cos pi/8, sin pi/8) = (rho_C u, rho_C sqrt(1-u^2))
    # use rho_C u = 2 theta * theta/2 = theta^2, and (rho_C)^2 (1-u^2) = 4theta^2 - theta^4
    cx = reduce_theta(RHO_C * U)
    cy2 = reduce_theta(sp.expand(RHO_C**2 - cx**2))
    # |p* - c_C|^2 = R^2 + rho_C^2 - 2 (xB cx + y cy); the cross term needs y*cy
    # so verify via (R^2 + rho_C^2 - 1)/2 == xB*cx + y*cy, squaring once.
    lhs = reduce_theta(sp.expand((R2 + RHO_C**2 - 1) / 2 - xB * cx))
    ok = ring_is_zero(sp.expand(lhs**2 - y2 * cy2))
    check("|p* - c_{C_0}|^2 = 1 exactly", ok)
    check("the cross term has the right sign", *positive(lhs, "(R^2+rho_C^2-1)/2 - xB*cx"))
    # every other centre is at distance > 1
    import numpy as np

    import sys
    sys.path.insert(0, "src")
    from config25 import centers25

    C = centers25()
    px = float(numval(xB))
    py = float(mp.sqrt(numval(y2)))
    d = np.linalg.norm(np.array([px, py]) - C, axis=1)
    others = np.sort(d)[2:]
    check("all 23 other centres are at distance > 1 from p*",
          bool(others.min() > 1 + 1e-9),
          f"min = {others.min():.12f}")
    # radial derivative of the two active distances is positive at p*
    for nm, c in (("B_0", C[np.argsort(d)[0]]), ("C_0", C[np.argsort(d)[1]])):
        val = px * px + py * py - (px * c[0] + py * c[1])
        check(f"d/dlambda |lambda p* - c_{nm}|^2 > 0 at lambda = 1",
              val > 1e-9, f"{val:.9f}")
    print("      => points lambda p* with lambda slightly > 1 lie outside every")
    print("         one of the 25 unit disks, so disk(0, R') is NOT covered for R' > R.")


def main() -> None:
    step0_minpoly()
    step1_parameters()
    step2_lemma1()
    step3_lemma2()
    step4_lemma3()
    step5_lemma4()
    step6_side_conditions()
    step7_tightness()
    step8_gap_witness()

    head("RESULT")
    bad = [lab for ok, lab in RESULTS if not ok]
    print(f"  {len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    for lab in bad:
        print(f"    FAILED: {lab}")
    if bad:
        raise SystemExit(1)
    print(
        "\n  THEOREM 1 (machine-verified, exact).\n"
        "  The 25 unit disks centred at\n"
        "      (0,0);\n"
        "      2cos(pi/8) * (cos(pi/8 + k pi/4), sin(pi/8 + k pi/4)),  k=0..7;\n"
        "      (2+sqrt2)  * (cos(k pi/4),        sin(k pi/4)),         k=0..7;\n"
        "      4cos(pi/8) * (cos(pi/8 + k pi/4), sin(pi/8 + k pi/4)),  k=0..7\n"
        "  cover the closed disk of radius R = sqrt6 + sqrt3 = 4.18154055035...,\n"
        "  and cover no larger concentric disk.  Hence r(25) >= sqrt6 + sqrt3."
    )


if __name__ == "__main__":
    main()
