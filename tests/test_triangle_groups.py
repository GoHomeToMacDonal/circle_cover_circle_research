import math

import numpy as np

from partition_search import LAMBDA
from triangle_groups import (
    TriangleParameters,
    known_centers,
    triangle_points,
    trial_parameters,
    validate_locking_groups,
)


def test_all_25_triangles_recover_intended_circles():
    parameters = trial_parameters(0)
    diagnostics = validate_locking_groups(parameters)
    points, labels, groups = triangle_points(parameters)
    centers, circle_labels, _ = known_centers()
    assert len(points) == len(labels) == 75
    assert len(groups) == len(centers) == len(circle_labels) == 25
    assert diagnostics["min_acute_margin"] > 0
    for center, group in zip(centers, groups):
        distances = np.linalg.norm(points[group] - center, axis=1)
        assert np.allclose(distances, LAMBDA, atol=1e-12)


def test_admissible_boundary_angles_keep_points_in_unit_disk():
    parameters = TriangleParameters(
        beta_o=math.pi / 3,
        beta_a=math.pi / 3,
        beta_b=math.pi / 4,
        beta_c=3 * math.pi / 8,
        phase_o=0.0,
    )
    points, _, _ = triangle_points(parameters)
    assert np.linalg.norm(points, axis=1).max() <= 1.0 + 1e-12
    validate_locking_groups(parameters)
