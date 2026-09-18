import math

import numpy as np

from config25 import LABELS, R25, centers25
from private_region_family import LAMBDA, boundary_arcs, diagnostics, family, owner_vertices


def test_family_has_expected_owner_groups_and_multiplicities():
    points, owners, names, groups = family(0.99)
    assert len(points) == len(owners) == len(names) == 128
    assert len(groups) == 25
    assert [len(group) for group in groups] == [8] + [6] * 8 + [5] * 8 + [4] * 8
    assert len(owner_vertices()) == 25


def test_rho_one_collapses_to_the_48_tight_geometric_points():
    points, _, _, _ = family(1.0)
    distinct = {(round(float(x), 10), round(float(y), 10)) for x, y in points}
    assert len(distinct) == 48


def test_every_group_has_owner_circle_as_its_mec():
    rho = 0.937
    points, _, _, groups = family(rho)
    for center, group in zip(centers25(), groups):
        vectors = points[group] - center
        assert np.allclose(np.linalg.norm(vectors, axis=1), rho, atol=2e-15)
        angles = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]) % (2 * math.pi))
        max_gap = np.diff(np.r_[angles, angles[0] + 2 * math.pi]).max()
        # No open semicircle contains the group, hence its MEC radius is rho.
        assert max_gap <= math.pi + 2e-14


def test_normalized_family_stays_in_unit_disk_for_entire_parameter_range():
    # Convexity gives the all-rho proof: each point is a convex combination of
    # an owner centre and a tight point, both in disk(0,R25).  Samples protect
    # the implementation and include both limiting ends.
    assert np.linalg.norm(centers25(), axis=1).max() < R25
    for rho in (1e-9, 0.5, 0.99, 0.999999, 1.0):
        points, _, _, _ = family(rho, normalized=True)
        assert np.linalg.norm(points, axis=1).max() <= 1.0 + 2e-14
        report = diagnostics(rho)
        assert report["max_owner_circle_error"] < 2e-14
    assert math.isclose(LAMBDA, 1.0 / R25)


def test_private_boundary_arc_combinatorics_matches_vertex_cycles():
    vertices = owner_vertices()
    arcs = boundary_arcs()
    assert set(vertices) == set(arcs)
    assert all(len(vertices[owner]) == len(arcs[owner]) for owner in vertices)
    assert arcs["A0"] == ("O", "A1", "B1", "C0", "B0", "A7")
    assert arcs["B0"] == ("A0", "C0", "S_R", "C7", "A7")
    assert arcs["C0"] == ("A0", "B1", "S_R", "B0")


def test_boundary_dense_family_is_nested_and_level_zero_is_unchanged():
    from private_region_family import boundary_dense_family

    base = family(0.99)
    dense0 = boundary_dense_family(0.99, 0)
    assert np.array_equal(base[0], dense0[0])
    assert base[1:] == dense0[1:]

    previous = set()
    expected_counts = (128, 304, 592, 1168)
    for level, expected in enumerate(expected_counts):
        points, owners, _, _ = boundary_dense_family(0.99, level)
        tagged = {(round(float(x), 11), round(float(y), 11), owner)
                  for (x, y), owner in zip(points, owners)}
        assert len(points) == len(tagged) == expected
        assert previous <= tagged
        previous = tagged


def test_boundary_dense_points_stay_on_owner_circles_and_in_target_disk():
    from private_region_family import boundary_dense_family

    for rho in (0.99, 0.999999, 1.0):
        points, _, _, groups = boundary_dense_family(rho, 2)
        assert np.linalg.norm(points, axis=1).max() <= R25 + 3e-13
        for center, group in zip(centers25(), groups):
            assert np.allclose(np.linalg.norm(points[group] - center, axis=1),
                               rho, atol=3e-13)
        normalized, *_ = boundary_dense_family(rho, 2, normalized=True)
        assert np.linalg.norm(normalized, axis=1).max() <= 1.0 + 1e-13


def test_rho_one_reuses_existing_target_intersection_anchors():
    from private_region_family import boundary_dense_family

    assert len(boundary_dense_family(1.0, 1)[0]) == 240
    assert len(boundary_dense_family(1.0, 2)[0]) == 464


def test_boundary_core_shape_is_a_homothetic_owner_copy():
    from private_region_family import boundary_dense_family, boundary_with_core_family

    rho = 0.99
    epsilon = 0.05
    outer, outer_owners, outer_names, outer_groups = boundary_dense_family(rho, 1)
    points, owners, names, groups = boundary_with_core_family(rho, 1, epsilon)
    assert len(points) == 2 * len(outer) == 608
    assert points[:len(outer)].tolist() == outer.tolist()
    assert owners[:len(outer)] == outer_owners
    assert names[:len(outer)] == outer_names
    center_by_owner = dict(zip(
        ["O"] + [f"A{k}" for k in range(8)] + [f"B{k}" for k in range(8)] +
        [f"C{k}" for k in range(8)], centers25()
    ))
    for i, (point, owner) in enumerate(zip(outer, outer_owners)):
        expected = center_by_owner[owner] + epsilon * (point - center_by_owner[owner])
        assert np.allclose(points[len(outer) + i], expected, atol=2e-15)
        assert names[len(outer) + i].startswith("core:")
    for center, group, outer_group in zip(centers25(), groups, outer_groups):
        distances = np.linalg.norm(points[group] - center, axis=1)
        assert np.count_nonzero(np.isclose(distances, rho, atol=3e-13)) == len(outer_group)
        assert np.count_nonzero(np.isclose(distances, epsilon * rho, atol=3e-13)) == len(outer_group)


def test_boundary_core_baseline_nesting_and_unit_disk_constraints():
    from private_region_family import boundary_dense_family, boundary_with_core_family

    dense = boundary_dense_family(0.999999, 2)
    baseline = boundary_with_core_family(0.999999, 2, 0.0)
    assert np.array_equal(dense[0], baseline[0])
    assert dense[1:] == baseline[1:]

    previous = set()
    for level, expected in enumerate((256, 608, 1184, 2336)):
        points, owners, _, _ = boundary_with_core_family(0.99, level, 0.1)
        tagged = {(round(float(x), 11), round(float(y), 11), owner)
                  for (x, y), owner in zip(points, owners)}
        assert len(points) == len(tagged) == expected
        assert previous <= tagged
        assert np.linalg.norm(points, axis=1).max() <= R25 + 3e-13
        previous = tagged

    normalized, *_ = boundary_with_core_family(0.999999, 2, 0.2,
                                                normalized=True)
    assert np.linalg.norm(normalized, axis=1).max() <= 1.0 + 1e-13


def test_boundary_core_scale_validation():
    import pytest
    from private_region_family import boundary_with_core_family

    with pytest.raises(ValueError):
        boundary_with_core_family(0.99, 0, -0.01)
    with pytest.raises(ValueError):
        boundary_with_core_family(0.99, 0, 1.0)


def test_multiple_radial_layers_form_aligned_spokes_and_expected_sizes():
    from private_region_family import (
        boundary_dense_family,
        boundary_with_radial_layers_family,
    )

    rho = 0.99
    outer, outer_owners, outer_names, _ = boundary_dense_family(rho, 0)
    small = (0.02, 0.05, 0.1)
    geometric = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7)
    uniform = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert len(boundary_with_radial_layers_family(rho, 0, small)[0]) == 512
    points, owners, names, groups = boundary_with_radial_layers_family(
        rho, 0, geometric
    )
    assert len(points) == 921
    assert len(boundary_with_radial_layers_family(rho, 0, uniform)[0]) == 1305
    assert len(points) == len(owners) == len(names)
    assert len(groups) == 25

    center_by_owner = dict(zip(LABELS, centers25()))
    outer_count = len(outer)
    positive = tuple(scale for scale in geometric if scale > 0)
    for outer_index, (point, owner) in enumerate(zip(outer, outer_owners)):
        center = center_by_owner[owner]
        direction = point - center
        for layer_index, scale in enumerate(positive):
            index = outer_count + 25 + layer_index * outer_count + outer_index
            assert np.allclose(points[index], center + scale * direction,
                               atol=3e-15)
            assert owners[index] == owner
            assert names[index].startswith(f"layer:{scale:.12g}:")


def test_radial_layer_schedules_are_nested_and_stay_in_target_disk():
    from private_region_family import boundary_with_radial_layers_family

    small = (0.02, 0.05, 0.1)
    geometric = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7)
    point_sets = []
    for scales in (small, geometric):
        points, owners, _, _ = boundary_with_radial_layers_family(0.999999, 0, scales)
        assert np.linalg.norm(points, axis=1).max() <= R25 + 3e-13
        point_sets.append({(round(float(x), 11), round(float(y), 11), owner)
                           for (x, y), owner in zip(points, owners)})
    assert point_sets[0] <= point_sets[1]
    normalized, *_ = boundary_with_radial_layers_family(
        0.999999, 0, geometric, normalized=True
    )
    assert np.linalg.norm(normalized, axis=1).max() <= 1.0 + 1e-13


def test_radial_layer_scale_validation_and_empty_baseline():
    import pytest
    from private_region_family import (
        boundary_dense_family,
        boundary_with_radial_layers_family,
    )

    base = boundary_dense_family(0.99, 1)
    empty = boundary_with_radial_layers_family(0.99, 1, ())
    assert np.array_equal(base[0], empty[0])
    assert base[1:] == empty[1:]
    for bad in ((0.1, 0.1), (-0.1,), (1.0,)):
        with pytest.raises(ValueError):
            boundary_with_radial_layers_family(0.99, 0, bad)
