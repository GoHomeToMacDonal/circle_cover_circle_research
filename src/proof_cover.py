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
Sign assertions use exact rational interval evaluation with a recorded margin;
non-ring algebraic signs use a rational isolating interval for their minimal
polynomial. Decimal high precision is used only to select/display an already
isolated algebraic branch, never as a proof threshold.

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

# Decimal evaluation is retained only for human-readable display.  Strict sign
# decisions use the rational enclosure or an algebraic isolating interval.
TH_NUM = mp.sqrt(2 + mp.sqrt(2))
# Exact rational enclosure for the distinguished real root theta.  All ring
# sign decisions below use this enclosure, never TH_NUM or a tolerance.
THETA_LO = sp.Rational("1.84775906502257351225636637879357657364483325172728")
THETA_HI = sp.Rational("1.84775906502257351225636637879357657364483325172730")

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
    """Reduce a rational expression modulo the theta minimal polynomial."""
    e = sp.expand(sp.together(expr))
    num, den = sp.fraction(sp.cancel(e))
    m = sp.Poly(MINPOLY.subs(x, th), th, domain="EX")
    num_poly = sp.Poly(sp.expand(num), th, domain="EX")
    den_poly = sp.Poly(sp.expand(den), th, domain="EX")
    if den_poly.degree() == 0:
        return sp.expand(num_poly.rem(m).as_expr() / den)
    inverse = sp.invert(den_poly, m)
    return sp.expand((num_poly * inverse).rem(m).as_expr())


def ring_is_zero(expr) -> bool:
    """Decide whether an element of Z[theta][s,t] is identically zero."""
    r = reduce_theta(expr)
    return sp.expand(r) == 0


def _interval_add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _interval_mul(a, b):
    vals = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(vals), max(vals)


def theta_interval(expr):
    """Evaluate a reduced ring element in an exact rational interval."""
    reduced = reduce_theta(expr)
    if reduced == 0:
        return sp.Integer(0), sp.Integer(0)
    if reduced.free_symbols - {th}:
        raise ValueError(f"sign expression has free symbols: {reduced.free_symbols}")
    poly = sp.Poly(reduced, th)
    value = (sp.Integer(0), sp.Integer(0))
    theta_box = (THETA_LO, THETA_HI)
    for coeff in poly.all_coeffs():
        value = _interval_add(_interval_mul(value, theta_box), (coeff, coeff))
    return value


def _fmt_interval(interval):
    lo, hi = interval
    return f"[{sp.N(lo, 18)}, {sp.N(hi, 18)}]"


def numval(expr):
    """200-digit value for display only; never used for a decision."""
    f = sp.lambdify(th, reduce_theta(expr), modules="mpmath")
    return f(TH_NUM)


def positive(expr, name: str) -> tuple[bool, str]:
    interval = theta_interval(expr)
    lo, hi = interval
    ok = lo > 0
    margin = lo if ok else min(abs(lo), abs(hi))
    return ok, f"{name} in {_fmt_interval(interval)}; certified margin >= {sp.N(margin, 18)}"


def nonneg(expr, name: str) -> tuple[bool, str]:
    """Exact non-strict sign using rational interval evaluation."""
    interval = theta_interval(expr)
    lo, hi = interval
    ok = lo >= 0
    if lo == 0 and hi == 0:
        note = f"{name} = 0 exactly (certified nonnegative margin 0)"
    else:
        margin = lo if ok else min(abs(lo), abs(hi))
        note = f"{name} in {_fmt_interval(interval)}; certified margin >= {sp.N(margin, 18)}"
    return ok, note


def _sqrt_rational_bounds(q, digits=90):
    """Certified rational bounds for the principal square root of q >= 0."""
    q = sp.Rational(q)
    if q < 0:
        raise ValueError(f"sqrt of negative rational {q}")
    if q == 0:
        return sp.Integer(0), sp.Integer(0)
    scale = 10 ** digits
    n = (q.p * scale * scale) // q.q
    k = sp.integer_nthroot(n, 2)[0]
    return sp.Rational(k, scale), sp.Rational(k + 1, scale)


def _interval_add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _interval_mul(a, b):
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(values), max(values)


def _principal_sqrt_interval(expr, digits=90):
    """Evaluate nested principal square roots using rational interval propagation."""
    expr = sp.cancel(sp.expand(expr))
    if expr.is_Rational:
        return sp.Rational(expr), sp.Rational(expr)
    if expr.is_Add:
        result = (sp.Integer(0), sp.Integer(0))
        for arg in expr.args:
            result = _interval_add(result, _principal_sqrt_interval(arg, digits))
        return result
    if expr.is_Mul:
        result = (sp.Integer(1), sp.Integer(1))
        for arg in expr.args:
            result = _interval_mul(result, _principal_sqrt_interval(arg, digits))
        return result
    if expr.is_Pow:
        base, exponent = expr.args
        if exponent.is_Rational and exponent.q == 1:
            n = int(exponent.p)
            if n == 0:
                return sp.Integer(1), sp.Integer(1)
            base_interval = _principal_sqrt_interval(base, digits)
            if n < 0:
                if base_interval[0] <= 0 <= base_interval[1]:
                    raise ValueError(f"reciprocal interval crosses zero: {expr}")
                return _principal_sqrt_interval(1 / base, digits)
            result = (sp.Integer(1), sp.Integer(1))
            for _ in range(n):
                result = _interval_mul(result, base_interval)
            return result
        if exponent == sp.Rational(1, 2):
            lo, hi = _principal_sqrt_interval(base, digits)
            if lo < 0:
                if sp.simplify(base) == 0:
                    lo = sp.Integer(0)
                else:
                    raise ValueError(f"principal square-root interval crossed zero: {expr}; {lo, hi}")
            return _sqrt_rational_bounds(lo, digits)[0], _sqrt_rational_bounds(hi, digits)[1]
    if expr.is_Number:
        return sp.Rational(expr), sp.Rational(expr)
    raise TypeError(f"unsupported algebraic expression: {expr}")


def algebraic_interval(expr):
    """Return a rational interval by propagating the original principal radicals.

    In particular, no decimal approximation is used to choose a conjugate root.
    The distinguished theta branch is substituted by its already certified
    rational interval, and every remaining square root is the principal one.
    """
    e = sp.cancel(sp.expand(expr)).subs(th, sp.sqrt(2 + sp.sqrt(2)))
    e = sp.radsimp(sp.simplify(e))
    if e == 0:
        return sp.Integer(0), sp.Integer(0)
    for digits in (70, 100, 140):
        try:
            # Keep the certified positive branch explicit.  The interval of
            # sqrt(2 + sqrt(2)) is propagated from rational endpoint bounds.
            return _principal_sqrt_interval(e, digits)
        except ValueError:
            continue
    raise ValueError(f"could not certify principal-radical interval for {e}")


def algebraic_positive(expr, name: str) -> tuple[bool, str]:
    interval = algebraic_interval(expr)
    lo, hi = interval
    ok = lo > 0
    return ok, f"{name} in {_fmt_interval(interval)}; certified margin >= {sp.N(lo if ok else 0, 18)}"


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
    poly = sp.Poly(MINPOLY, x)
    lo_sign = sp.sign(poly.eval(THETA_LO))
    hi_sign = sp.sign(poly.eval(THETA_HI))
    check("theta is isolated by the rational interval [THETA_LO, THETA_HI]",
          lo_sign == -1 and hi_sign == 1 and poly.count_roots(THETA_LO, THETA_HI) == 1,
          f"width = {sp.N(THETA_HI - THETA_LO, 8)}; endpoint signs = ({lo_sign}, {hi_sign})")
    check("sqrt2 = theta^2 - 2",
          sp.simplify(sp.sqrt(2) - (exact**2 - 2)) == 0)
    count = sp.Poly(MINPOLY, x).count_roots(sp.Rational(9, 5), sp.Rational(19, 10))
    check("exactly one root of MINPOLY lies in (1.8, 1.9)", count == 1,
          "exact rational root count on (9/5, 19/10)")


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


def exact_centers25():
    q = sp.sqrt(2)
    rotations = ((1, 0), (q / 2, q / 2), (0, 1), (-q / 2, q / 2),
                 (-1, 0), (-q / 2, -q / 2), (0, -1), (q / 2, -q / 2))

    def rotate(v, k):
        c, ss = rotations[k % 8]
        return (sp.expand(c * v[0] - ss * v[1]),
                sp.expand(ss * v[0] + c * v[1]))

    q2 = sp.sqrt(2)
    a = (1 + q2 / 2, q2 / 2)
    b = (2 + q2, 0)
    c = (2 + q2, q2)
    result = [(sp.Integer(0), sp.Integer(0))]
    result += [rotate(a, k) for k in range(8)]
    result += [rotate(b, k) for k in range(8)]
    result += [rotate(c, k) for k in range(8)]
    return result


def step8_gap_witness() -> None:
    head("Step 4.  The gap that opens beyond R is not filled by any other disk")
    # p* = the outer intersection of circles B_0 and C_0; |p*| = R exactly.
    # Exact coordinates: p* is on S_R at angle alpha_B(R) from the B_0 axis.
    cB = sp.expand((R2 + RHO_B**2 - 1)) / (2 * RHO_B)
    xB = reduce_theta(sp.together(cB))
    y2 = reduce_theta(sp.expand(R2 - xB**2))
    yB = sp.sqrt(y2)
    check("|p*|^2 = R^2 by construction", ring_is_zero(xB**2 + y2 - R2))
    check("y(p*)^2 > 0 so p* is off the symmetry axis", *positive(y2, "y^2"))
    print(f"      x(p*) = {mp.nstr(numval(xB), 20)},  y(p*) = {mp.nstr(mp.sqrt(numval(y2)), 20)}")
    d2B = sp.expand((xB - RHO_B) ** 2) + y2
    check("|p* - c_{B_0}|^2 = 1 exactly", ring_is_zero(d2B - 1))
    cx = reduce_theta(RHO_C * U)
    cy2 = reduce_theta(sp.expand(RHO_C**2 - cx**2))
    lhs = reduce_theta(sp.expand((R2 + RHO_C**2 - 1) / 2 - xB * cx))
    c0_exact_ok = ring_is_zero(sp.expand(lhs**2 - y2 * cy2))
    check("|p* - c_{C_0}|^2 = 1 exactly", c0_exact_ok)
    check("the cross term has the right sign", *positive(lhs, "(R^2+rho_C^2-1)/2 - xB*cx"))

    # The following are algebraic interval decisions.  No floating nearest-site
    # sort or tolerance is used: the two active sites are exact equalities, and
    # every other site has an isolated positive distance-squared margin.
    centers = exact_centers25()
    labels = ["O"] + [f"A{k}" for k in range(8)] + [f"B{k}" for k in range(8)] + [f"C{k}" for k in range(8)]
    p = (xB, yB)
    nonactive_margins = []
    for label, center in zip(labels, centers):
        d2 = sp.expand((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2 - 1)
        if label == "B0":
            check(f"active site {label} has distance exactly 1", ring_is_zero(d2B - 1))
        elif label == "C0":
            check(f"active site {label} has distance exactly 1", c0_exact_ok)
        else:
            ok, note = algebraic_positive(d2, f"|p* - c_{label}|^2 - 1")
            nonactive_margins.append((label, note))
            check(f"nonactive site {label} is strictly outside p*", ok, note)
    print(f"      minimum nonactive margin is certified separately for {len(nonactive_margins)} sites")

    for nm, center in (("B_0", centers[9]), ("C_0", centers[17])):
        derivative = sp.expand(R2 - (p[0] * center[0] + p[1] * center[1]))
        check(f"d/dlambda |lambda p* - c_{nm}|^2 > 0 at lambda = 1",
              *algebraic_positive(derivative, f"radial derivative half-value for {nm}"))
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
