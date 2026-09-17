import math

import numpy as np

from partition_search import LAMBDA, PartitionSolver, candidate_masks, generate_candidate


def test_generated_candidates_are_reproducible_and_inside_unit_disk():
    a = generate_candidate(7)
    b = generate_candidate(7)
    assert np.array_equal(a, b)
    assert len(a) > 80
    assert np.linalg.norm(a, axis=1).max() <= 1.0 + 1e-12
    # Rotation by pi/4 preserves the point set.
    rotation = np.array([[math.sqrt(0.5), -math.sqrt(0.5)],
                         [math.sqrt(0.5), math.sqrt(0.5)]])
    keys = {(round(x, 10), round(y, 10)) for x, y in a}
    assert all((round(x, 10), round(y, 10)) in keys for x, y in a @ rotation.T)


def test_candidate_masks_and_solver_on_separated_points():
    points = np.array([[-0.8, 0.0], [0.0, 0.0], [0.8, 0.0]])
    masks, _ = candidate_masks(points, radius=0.3)
    assert {m.bit_count() for m in masks} == {1}
    solver = PartitionSolver(points, masks, radius=0.3)
    result = solver.decide(target=2)
    assert result.status == "infeasible"
    assert result.lower_bound == 3


def test_close_points_form_one_group():
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.05, 0.02]])
    masks, _ = candidate_masks(points, radius=LAMBDA)
    solver = PartitionSolver(points, masks)
    result = solver.decide(target=1)
    assert result.status == "feasible"
    assert len(result.solution_masks) == 1
    assert result.solution_masks[0] == 0b111


def test_cover_and_partition_optima_are_represented_by_masks():
    # Two close pairs separated by more than 2r require exactly two groups.
    points = np.array([[-0.6, 0.0], [-0.55, 0.02], [0.55, 0.0], [0.6, -0.02]])
    masks, _ = candidate_masks(points, radius=0.2)
    solver = PartitionSolver(points, masks, radius=0.2)
    assert solver.decide(target=1).status == "infeasible"
    result = PartitionSolver(points, masks, radius=0.2).decide(target=2)
    assert result.status == "feasible"
    assert len(result.solution_masks) == 2
