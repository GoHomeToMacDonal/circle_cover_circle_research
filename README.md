# Circles covering circles: the case n = 25

Target problem: Erich Friedman's *Circles Covering Circles*
(<https://erich-friedman.github.io/packing/circovcir/>).  For n = 25 the best
known covering radius is

```
r(25) = sqrt6 + sqrt3 = sqrt3 (1 + sqrt2) = 4.181540550352056...
```

found numerically by Jeremy Tan in 2018.  No proof of optimality is known.  This
repository (a) determines the configuration exactly and proves that it covers
exactly that disk, and (b) proves several rigorous partial optimality results.

Reproduce:

```
uv run pytest -q                        # regression tests
uv run python src/proof_cover.py        # Theorem 1: 43 exact symbolic checks
uv run python src/measure_bound.py      # Theorem 2: certified upper bound
uv run python src/symmetry_bb.py 4.19   # Theorem 3: symmetric-class B&B
uv run python src/search.py 8           # unrestricted numerical search
```

## What is proved

| # | statement | method |
|---|---|---|
| 1 | The explicit 25-disk configuration below covers **exactly** the disk of radius sqrt6+sqrt3 and no larger concentric disk. Hence **r(25) >= sqrt6 + sqrt3**. | exact symbolic algebra in `Z[theta]/(theta^4-4theta^2+2)`, 43/43 checks (`src/proof_cover.py`) |
| 2 | **r(25) <= 4.362226** | certified fractional-covering (measure / "probabilistic") LP (`src/measure_bound.py`) |
| 3 | No configuration with k-fold rotational symmetry covers a disk of radius 4.25, for every k in {5, 6, 8, 12, 24, 25}; radius 4.19 for k in {6, 8, 12, 24, 25}; radius **4.183** for k = 8, the class containing the record. | interval branch and bound with exact deepest-hole evaluation (`src/symmetry_bb.py`) |
| 4 | Rigorous upper bounds for the unproved intermediate cases, e.g. r(11) <= 2.6966, r(15) <= 3.2547, r(20) <= 3.8749, r(24) <= 4.2734. | bootstrapped radial programme (`src/bootstrap_bound.py`) |

**Still open: r(25) = sqrt6 + sqrt3.**  What is proved is
`4.181540550 <= r(25) <= 4.362226`, plus optimality inside every reasonably
symmetric class, plus (statement 3 with k = 8) optimality within 0.0015 inside the
record's own symmetry class.

## The configuration (Theorem 1)

With `u = cos(pi/8)`, so that `theta := 2u = sqrt(2 + sqrt2)` is the root of
`x^4 - 4x^2 + 2` in (1.8, 1.9), the 25 centres are

```
O   : (0, 0)
A_k : rho_A = 2u        = 1.847759065...   at angle pi/8 + k pi/4,  k = 0..7
B_k : rho_B = 4u^2      = 2 + sqrt2        at angle       k pi/4,   k = 0..7
C_k : rho_C = 4u        = 3.695518130...   at angle pi/8 + k pi/4,  k = 0..7
```

Symmetry group: the dihedral group D_8 of order 16.  Three exact identities
organise everything:

```
rho_B = rho_A^2 ,      rho_C = 2 rho_A ,      rho_B - 1 = 1 + sqrt2 = r(9)
```

The last one is the pretty one: **{O} together with ring A is exactly the
proved-optimal n = 9 configuration**, which covers the disk of radius 1 + sqrt2,
and the innermost points of the eight B-disks land precisely on that disk's
boundary.  Correspondingly `R = sqrt3 * r(9)`.

### Proof

A fundamental domain for D_8 is the wedge `W = {0 <= arg p <= pi/8}`.  A unit disk
at distance rho from the origin meets the circle `S_t = {|p| = t}` in an arc of
half-width `alpha(t,rho) = arccos((t^2 + rho^2 - 1)/(2 t rho))`.  Only four disks
matter inside W: O at the origin, B_0 centred on the edge `arg = 0`, and A_0, C_0
centred on the edge `arg = pi/8`.  So covering W reduces to four one-variable
statements, and in the variable `s = t^2` they are all **quadratic**:

```
t in [0, 1]          O alone covers S_t
t in [1, 1+sqrt2]    t^2 + rho_A^2 - 1 - 2 u rho_A t = (t-1)(t-(1+sqrt2)) <= 0
                     <=> alpha_A >= pi/8, so A_0 alone spans the wedge
t in [1+sqrt2, tx]   N_{A,B}(s) = 4u^2 (s - (3+2sqrt2))(s - (5+2sqrt2)) <= 0
t in [tx, R]         N_{C,B}(s) = 2    (s - (5+2sqrt2))(s - (9+6sqrt2)) <= 0
```

Here `N_{p,q}(s) <= 0` is *equivalent* to `alpha_p + alpha_q >= pi/8` (obtained by
`cos(alpha_p + alpha_q) <= cos(pi/8)` and clearing denominators), and

```
tx  = sqrt(5 + 2 sqrt2) = 2.797932651931813 = sqrt(2 rho_A^2 + 1)
      the radius at which alpha_A = alpha_C, forced by rho_C = 2 rho_A
R^2 = 9 + 6 sqrt2
```

Each quadratic vanishes exactly at the two endpoints of its own interval, so the
four phases chain together with no gap and with **zero slack at every join**.

Tightness beyond R uses the explicit witness

```
p* = (2 + 3 sqrt2 / 2,  sqrt2 / 2),        |p*|^2 = 9 + 6 sqrt2 = R^2
```

at distance exactly 1 from B_0 and C_0 and exactly sqrt5 from the next nearest
centre, with both active distances strictly increasing along the ray through it;
so `lambda p*` is uncovered for every `lambda > 1`.

`src/proof_cover.py` verifies all of this by reducing polynomials modulo
`x^4 - 4x^2 + 2`, which is a decision procedure, so the identities are proved, not
estimated.  Sign assertions are certified by 200-digit evaluation with recorded
margins.

## How rigid the configuration is

* **48 tight points** (points of the disk covered with zero slack) in four D_8
  orbits: 8 at radius 1, 8 at radius 1+sqrt2, 16 at radius sqrt(5+2sqrt2), 16 on
  the boundary S_R.  All but the last orbit are *triple* points.
* **The boundary is covered with zero waste.**  On S_R the eight B-arcs have
  half-width `alpha_B(R) = 9.735610317 deg`, the eight C-arcs
  `alpha_C(R) = 12.764389683 deg`, and `alpha_B(R) + alpha_C(R) = pi/8` exactly, so
  the sixteen arcs **tile** S_R: they meet in 16 points and overlap nowhere.
* Every outward normal at a tight point, from each of its owning disks, points in
  an exact multiple of 45 degrees.
* The purely radial necessary condition `sum_i alpha(t, rho_i) >= pi` holds with
  **equality** at `t = 1 + sqrt2` and at `t = R`.

## Theorem 2: the certified upper bound r(25) <= 4.362226

This is the rigorous form of the "probabilistic method".  If mu is a measure on
disk(0,R) with `mu(B(x,1)) <= 1` for every x, then any covering by N unit disks
gives `mu(disk(0,R)) <= N`.  The supremum of `mu(disk(0,R))` is the *fractional
covering number* phi(R) -- the exact strength of the whole family of measure and
probabilistic arguments.

Since the constraints and the objective are rotation invariant and the feasible set
is convex, averaging over rotations shows phi(R) is attained by a rotation
invariant measure.  So phi(R) is the value of a **one-dimensional** LP: place mass
w_k on the circle of radius t_k; the constraint at |x| = sigma reads
`sum_k w_k alpha(t_k, sigma)/pi <= 1`.  Solving on a grid and then certifying
feasibility on a much finer grid (rescaling by the observed maximum load):

```
      R        phi(R) certified
 4.000000         20.98909
 4.181541         22.88447    <-- sqrt6+sqrt3; integrality gap about 2.1
 4.300000         24.34886
 4.362226         25.00000    <-- threshold
 4.500000         26.40591
```

Hence **r(25) <= 4.362226**.  For comparison the plain area bound (density 1/pi)
gives only 5, and Kershner's hexagonal density `2 pi / sqrt27` gives 4.547.

## Theorem 3: optimality inside symmetric classes

If 25 unit disks are invariant under rotation by `2 pi / k` about the origin, every
orbit has size k except for at most one orbit at the origin, so
`25 = k m + [centre]`:

| k | structure | free parameters |
|---|---|---|
| 25 | one orbit of 25 | 1 |
| 24 | centre + one orbit of 24 | 1 |
| 12 | centre + two orbits of 12 | 3 |
| 8 | centre + three orbits of 8 (**the record**) | 5 |
| 6 | centre + four orbits of 6 | 7 |
| 5 | five orbits of 5 | 9 |

(one angle is set to 0 by a global rotation, and the ring radii are taken
non-decreasing).  The branch and bound uses the exact deepest-hole function
`H(C,R) = max_{|p| <= R} min_i |p - c_i|` (so `disk(0,R)` is covered iff
`H <= 1`), evaluated without any discretisation from Voronoi vertices and boundary
stationary points (`src/hole.py`).  Since H is 1-Lipschitz in `max_i |c_i - c_i'|`,
a parameter box V with centre v0 is eliminated as soon as

```
H(C(v0), R) - delta(V) > 1,        delta(V) = max_j (h_j^rho + rho_j^max h_j^theta)
```

and boxes are additionally killed for free by the radial arc test (for some
`t <= R`, even the most generous radii in the box give total arc `< 2 pi` on S_t).

Results (all boxes eliminated, so *no* configuration in the class covers the disk):

```
   k  dim   R = 4.25          R = 4.20          R = 4.19          R = 4.183
  25    1   proved (1)        proved            proved            proved
  24    1   proved (2)        proved            proved            proved
  12    3   proved (2)        proved            proved            proved
   8    5   proved (7284)     proved (40306)    proved (80588)    proved (235604)
   6    7   proved (44372)    proved (85681)    proved (111392)   --
   5    9   proved (351735)   inconclusive      --                --
```

(node counts in parentheses).  Combined with Theorem 1: **inside the record's own
symmetry class the optimum lies in [4.181540550, 4.183]**, and no more symmetric
arrangement does better than 4.25.

## Theorem 4: rigorous bounds for the intermediate n

The radial programme (arc condition + nesting + proved values r(1..10) + the
measure bound, `src/radial_bound.py`, `src/bootstrap_bound.py`) yields rigorous
upper bounds for cases where nothing was proved before:

```
 n     upper bound   best known (conjectured)
11        2.6966          2.631
12        2.8186          2.769
13        2.9767          2.884
14        3.1082          3.014
15        3.2547          3.143
16        3.3567          3.244
17        3.4748          3.349
18        3.5746          3.446
19        3.7256          3.6055 = sqrt13
20        3.8749          3.692
21        3.9823          3.804
22        4.1031          3.948
23        4.1941          4.000
24        4.2734          4.076
25        4.3732          4.1815 = sqrt6+sqrt3
```

## What does not work, and why

* **Radial information alone caps out at about 4.36.**  A covering must satisfy
  `sum_i alpha(t, rho_i) >= pi` for all t, which uses only the 25 radial
  distances.  Its LP relaxation *is* the measure bound of Theorem 2, and keeping
  the count integral, adding the nesting condition
  `#{i : rho_i < d} >= nmin(d-1)`, and bootstrapping nmin recursively all fail to
  improve on 4.362226 (the discretisation loss of the integer programme even
  exceeds the integrality gain).  The remaining gap is genuinely two-dimensional.
* **The 48 tight points do not certify local optimality.**  If a larger disk were
  covered, rescaling gives 25 disks of radius rho < 1 covering disk(0,R*), so with
  `c_i = c_i* + w_i` each tight point p_j needs an owner i with `w_i . n_ij > 0`
  (else `|p_j - c_i| >= 1` identically).  By Gordan's theorem that is a finite 0/1
  feasibility problem, and `src/local_opt.py` finds it **feasible**.  So the
  first-order tight-point test is inconclusive; second-order or richer certificate
  sets are needed.
* **Sub-certificates do not add up.**  A 56-point set on and near S_R provably
  needs exactly 16 disks of radius `1/R*` (`src/outer_partition.py`,
  `docs/outer_partition_certificate.md`), and an inner 65-point set needs 9, but
  the union of 180 points still needs only 19: a single disk can serve points of
  both certificates, so `16 + 9 = 25` cannot be concluded.  Making the two regions
  geometrically separated (gap `> 2/R*`) costs so much radius that the sum drops to
  about 22.
* **Locking gadgets fail.**  Giving each of the 25 disks three points whose unique
  minimum enclosing circle is that disk does *not* force a competitor to use the
  same 25 disks: `docs/triangle_group_search.md` exhibits an explicit 15-disk
  recombination at radius `0.999999/R*`.
* **Finite point sets cannot give the exact value.**  If a finite
  `P subset disk(0,R*)` needs 26 disks of radius rho, one only gets
  `r(25) <= R*/rho`, which tends to R* but never reaches it.  An exact proof needs
  a finite certificate *plus* a stability / local-rigidity argument.

## Numerical evidence for optimality

`src/search.py` minimises `H(C,R)` by cutting planes plus Chebyshev alternation,
with H evaluated exactly (never on a net):

```
      R        min H found     covers?
 4.181541      1.000000000       yes
 4.181600      1.000014217        no
 4.185000      1.000827315        no
 4.190000      1.002023046        no
 4.200000      1.004414509        no
 4.250000      1.016371825        no
```

At R = 4.1816 the best value found, 1.0000142, is *exactly* what rigid rescaling
of the optimal configuration gives ((R - R*)/R* = 1.41e-5).  Local descent seeded
at the optimum achieves no improvement whatever over pure rescaling, which is what
one expects at a strict local maximum.  Unseeded restarts do not get below
H = 1.025 at R = R*.

## Files

| file | contents |
|---|---|
| `src/geom.py` | exact covering radius via the 1-D arc method |
| `src/hole.py` | exact deepest hole `H(C,R)` by Voronoi / boundary candidate enumeration |
| `src/extract_centers.py` | recovers the centres from Friedman's PNG (the PNGs carry an alpha channel and must be composited on white) |
| `src/config25.py` | the exact configuration, tight points, radial coverage profile |
| `src/wedge.py` | the four-phase wedge reduction, numerically |
| `src/proof_cover.py` | **Theorem 1**, exact symbolic proof |
| `src/measure_bound.py` | **Theorem 2**, certified fractional / measure bound |
| `src/symmetry_bb.py` | **Theorem 3**, interval branch and bound over symmetric classes |
| `src/arc_bound.py`, `src/radial_bound.py`, `src/bootstrap_bound.py` | **Theorem 4**, the radial programme and its bootstrap |
| `src/local_opt.py` | tight-point / normal-cone first-order test (inconclusive, by design) |
| `src/certificate.py` | finite point-set set-cover certificates (LP dual = measure, ILP = integral) |
| `src/outer_partition.py`, `src/partition_search.py` | the 56-point boundary certificate (needs exactly 16 disks) |
| `src/search.py` | unrestricted numerical search |
