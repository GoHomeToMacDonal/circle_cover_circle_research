import numpy as np

from config25 import centers25
from spoke_refinement import continuous_spoke_gaps, refined_points


def test_known_owner_centers_cover_every_continuous_spoke():
    gaps = continuous_spoke_gaps(0.99, 0, centers25(), 0.99)
    assert gaps == []


def test_refinement_adds_one_midpoint_per_uncovered_spoke_gap():
    source_case = {
        "rho": 0.99,
        "level": 0,
        "schedule": "none",
        "radial_scales": [],
        "radius_factor": 0.1,
        "radius": 0.099,
        "points": 128,
        "cover": {"solution_centers": [[100.0, 100.0]]},
    }
    points, gaps = refined_points(source_case)
    assert len(gaps) == 128
    assert len(points) == 256
    assert all(abs(float(gap["midpoint"]) - 0.5) < 1e-15 for gap in gaps)
    assert np.isfinite(points).all()
