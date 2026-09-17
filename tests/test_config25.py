"""Regression tests for the n = 25 results."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config25 import R25, RHO_A, RHO_B, RHO_C, centers25, dedup, tight_points
from geom import covering_radius
from hole import deepest_hole

R_EXACT = math.sqrt(6) + math.sqrt(3)


def test_parameters_closed_form():
    assert RHO_A == pytest.approx(2 * math.cos(math.pi / 8), abs=1e-15)
    assert RHO_B == pytest.approx(RHO_A**2, abs=1e-14)
    assert RHO_C == pytest.approx(2 * RHO_A, abs=1e-14)
    assert RHO_B - 1 == pytest.approx(1 + math.sqrt(2), abs=1e-14)
    assert R_EXACT == pytest.approx(math.sqrt(3) * (1 + math.sqrt(2)), abs=1e-14)
    assert R_EXACT**2 == pytest.approx(9 + 6 * math.sqrt(2), abs=1e-13)


def test_configuration_has_25_centers():
    assert centers25().shape == (25, 2)


def test_covering_radius_is_sqrt6_plus_sqrt3():
    assert covering_radius(centers25()) == pytest.approx(R_EXACT, abs=1e-9)


def test_deepest_hole_is_exactly_one_at_R():
    h, _ = deepest_hole(centers25(), R_EXACT)
    assert h == pytest.approx(1.0, abs=1e-11)


def test_not_covered_beyond_R():
    for d in (1e-5, 1e-4, 1e-3):
        h, _ = deepest_hole(centers25(), R_EXACT + d)
        assert h > 1.0


def test_boundary_arcs_tile_exactly():
    """On S_R the 8 B-arcs and 8 C-arcs partition the circle with zero overlap."""
    def alpha(t, rho):
        return math.acos((t * t + rho * rho - 1) / (2 * t * rho))

    aB = alpha(R_EXACT, RHO_B)
    aC = alpha(R_EXACT, RHO_C)
    assert 8 * (2 * aB + 2 * aC) == pytest.approx(2 * math.pi, abs=1e-13)
    assert aB + aC == pytest.approx(math.pi / 8, abs=1e-14)


def test_48_tight_points_in_four_orbits():
    C = centers25()
    P, _, _ = tight_points(C, R_EXACT, tol=1e-9)
    P = dedup(P, tol=1e-7)
    assert len(P) == 48
    radii = np.sort(np.hypot(P[:, 0], P[:, 1]))
    groups = {}
    for r in radii:
        key = round(float(r), 6)
        groups[key] = groups.get(key, 0) + 1
    assert sorted(groups.items()) == [
        (1.0, 8),
        (pytest.approx(1 + math.sqrt(2), abs=1e-6), 8),
        (pytest.approx(math.sqrt(5 + 2 * math.sqrt(2)), abs=1e-6), 16),
        (pytest.approx(R_EXACT, abs=1e-6), 16),
    ]


def test_witness_point_beyond_R():
    """p* = (2 + 3 sqrt2/2, sqrt2/2) is at distance 1 from B_0 and C_0 only."""
    p = np.array([2 + 1.5 * math.sqrt(2), math.sqrt(2) / 2])
    assert np.dot(p, p) == pytest.approx(9 + 6 * math.sqrt(2), abs=1e-12)
    C = centers25()
    d = np.sort(np.linalg.norm(p - C, axis=1))
    assert d[0] == pytest.approx(1.0, abs=1e-12)
    assert d[1] == pytest.approx(1.0, abs=1e-12)
    assert d[2] == pytest.approx(math.sqrt(5), abs=1e-12)


def test_arc_condition_tight_at_two_radii():
    """sum_i alpha(t, rho_i) = pi exactly at t = 1 + sqrt2 and at t = R."""
    from arc_bound import alpha

    rho = np.hypot(centers25()[:, 0], centers25()[:, 1])
    for t in (1 + math.sqrt(2), R_EXACT):
        assert float(alpha(t, rho).sum()) == pytest.approx(math.pi, abs=1e-12)
