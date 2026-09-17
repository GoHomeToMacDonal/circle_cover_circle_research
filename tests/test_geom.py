"""Sanity checks for geom.covering_radius against configurations whose covering
radius is known in closed form, plus cross-validation against a brute-force grid.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geom import RingConfig, covering_radius, uncovered_depth


def test_n1_single_circle():
    assert covering_radius(np.array([[0.0, 0.0]])) == pytest.approx(1.0, abs=1e-10)


def test_n3_optimum():
    best = max(
        covering_radius(RingConfig(k=3, rings=((rho, 0.0),), center=False).centers())
        for rho in np.linspace(0.3, 1.2, 2001)
    )
    assert best == pytest.approx(2 / math.sqrt(3), abs=1e-4)


def test_n4_optimum():
    best = max(
        covering_radius(RingConfig(k=4, rings=((rho, 0.0),), center=False).centers())
        for rho in np.linspace(0.3, 1.5, 2001)
    )
    assert best == pytest.approx(math.sqrt(2), abs=1e-4)


def test_n7_hexagonal_exact():
    """Centre + 6 circles at distance sqrt(3) covers exactly radius 2."""
    c = RingConfig(k=6, rings=((math.sqrt(3), 0.0),), center=True).centers()
    assert covering_radius(c) == pytest.approx(2.0, abs=1e-9)


def test_n9_closed_form():
    """Centre + 8 at rho = 2 cos(pi/8) covers exactly radius 1 + sqrt(2)."""
    c = RingConfig(k=8, rings=((2 * math.cos(math.pi / 8), math.pi / 8),)).centers()
    assert covering_radius(c) == pytest.approx(1 + math.sqrt(2), abs=1e-9)


def test_n9_is_a_local_max():
    base = 2 * math.cos(math.pi / 8)
    r0 = covering_radius(RingConfig(k=8, rings=((base, 0.0),)).centers())
    for d in (-3e-3, -1e-3, 1e-3, 3e-3):
        r = covering_radius(RingConfig(k=8, rings=((base + d, 0.0),)).centers())
        assert r <= r0 + 1e-12


@pytest.mark.parametrize("seed", range(20))
def test_random_configs_match_grid(seed: int):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-2.5, 2.5, size=(10, 2))
    R = covering_radius(c)
    if R < 0.05:
        pytest.skip("degenerate")
    assert uncovered_depth(c, R - 1e-6, m=500) <= 1e-6
    assert uncovered_depth(c, R + 0.05, m=500) > 0.0
