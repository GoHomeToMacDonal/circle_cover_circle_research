import numpy as np

from finite_cover import Budget, enumerate_masks, minimum_cover, verify_witness
from private_region_family import family


def test_tangent_pair_is_represented_by_its_unique_midpoint():
    points = np.array([[-1.0, 0.0], [1.0, 0.0]])
    candidates = enumerate_masks(points, 1.0, Budget(time_limit=2.0))
    assert candidates.status == "complete"
    assert candidates.masks == [0b11]
    assert np.allclose(candidates.centers[0b11], (0.0, 0.0))

    inside = np.array([[-1.0 + 5e-9, 0.0], [1.0 - 5e-9, 0.0]])
    assert enumerate_masks(inside, 1.0, Budget(time_limit=2.0)).masks == [0b11]
    outside = np.array([[-1.0 - 5e-9, 0.0], [1.0 + 5e-9, 0.0]])
    assert {mask.bit_count() for mask in
            enumerate_masks(outside, 1.0, Budget(time_limit=2.0)).masks} == {1}


def test_complete_small_instances_return_exact_optimum_and_witness():
    separated = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    family_result, result = minimum_cover(
        separated, 0.4, Budget(time_limit=2.0, max_nodes=10_000))
    assert family_result.status == "complete"
    assert result.status == "optimal"
    assert result.optimum == result.lower_bound == result.upper_bound == 3
    assert verify_witness(separated, result.solution_centers, 0.4)["covered"]

    square = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    _, result = minimum_cover(square, 0.71, Budget(time_limit=2.0))
    assert result.status == "optimal"
    assert result.optimum == 1


def test_generation_limits_return_unknown_not_false_infeasible():
    points = np.array([[float(i), 0.0] for i in range(5)])
    candidates = enumerate_masks(
        points, 1.0, Budget(time_limit=10.0, max_candidates=0))
    assert candidates.status == "unknown"
    assert candidates.stop_reason == "candidate_limit"
    assert candidates.masks == []


def test_node_limit_returns_valid_bounds_and_progress_events():
    points, *_ = family(0.99)
    events = []
    candidates, result = minimum_cover(
        points, 0.99,
        Budget(time_limit=5.0, max_nodes=0, progress_interval=0.001),
        events.append,
    )
    assert candidates.status == "complete"
    assert result.status == "unknown"
    assert result.stop_reason == "node_limit"
    assert result.nodes == 0
    assert 0 < result.lower_bound < result.upper_bound <= 25
    stages = {event["stage"] for event in events}
    assert "candidate_generation" in stages
    assert "candidate_generation_done" in stages
    assert "set_cover_start" in stages
    assert "set_cover_stopped" in stages


def test_empty_input_needs_no_circles():
    _, result = minimum_cover(np.empty((0, 2)), 1.0, Budget(time_limit=1.0))
    assert result.status == "optimal"
    assert result.optimum == 0


def test_global_deadline_is_checked_on_far_pair_continue_paths(monkeypatch):
    import finite_cover

    clock = [0.0]

    def fake_monotonic():
        clock[0] += 0.0001
        return clock[0]

    monkeypatch.setattr(finite_cover.time, "monotonic", fake_monotonic)
    points = np.array([[3.0 * i, 0.0] for i in range(50)])
    candidates, result = finite_cover.minimum_cover(
        points, 1.0, finite_cover.Budget(time_limit=0.01, max_nodes=100_000))
    assert candidates.status == "unknown"
    assert candidates.stop_reason == "time_limit"
    assert 0 < candidates.pairs_done < candidates.pairs_total
    assert result.status == "unknown"
    assert result.stop_reason == "time_limit"


def test_candidate_done_callback_cannot_cross_deadline_and_report_complete(monkeypatch):
    import finite_cover

    clock = [0.0]
    monkeypatch.setattr(finite_cover.time, "monotonic", lambda: clock[0])

    def callback(event):
        if event["stage"] == "candidate_generation_done":
            clock[0] = 2.0

    candidates = finite_cover.enumerate_masks(
        np.array([[0.0, 0.0]]), 1.0,
        finite_cover.Budget(time_limit=1.0), callback)
    assert candidates.status == "unknown"
    assert candidates.stop_reason == "time_limit"


def test_solver_done_callback_cannot_cross_deadline_and_report_optimal(monkeypatch):
    import finite_cover

    clock = [0.0]
    monkeypatch.setattr(finite_cover.time, "monotonic", lambda: clock[0])

    def callback(event):
        if event["stage"] == "set_cover_done":
            clock[0] = 2.0

    candidates, result = finite_cover.minimum_cover(
        np.array([[0.0, 0.0]]), 1.0,
        finite_cover.Budget(time_limit=1.0), callback)
    assert candidates.status == "complete"
    assert result.status == "unknown"
    assert result.stop_reason == "time_limit"
    assert result.optimum is None
