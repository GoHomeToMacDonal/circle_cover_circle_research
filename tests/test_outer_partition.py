from outer_partition import (
    compatibility_graph,
    explicit_partition,
    maximal_cliques,
    outer_points,
    verify,
)


def test_outer_point_set_and_maximal_cliques():
    points, labels = outer_points()
    assert len(points) == len(labels) == 56
    assert len(set(labels)) == 56
    cliques = maximal_cliques(compatibility_graph(points))
    sizes = [mask.bit_count() for mask in cliques]
    assert len(cliques) == 56
    assert sizes.count(3) == 8
    assert sizes.count(4) == 40
    assert sizes.count(5) == 8


def test_explicit_groups_partition_all_points():
    groups = explicit_partition()
    assert len(groups) == 16
    assert sorted(i for group in groups for i in group) == list(range(56))


def test_outer_certificate_proves_exactly_16_groups():
    certificate = verify()
    assert certificate["search_15"]["status"] == "infeasible"
    assert len(certificate["partition_16"]) == 16


def test_outer_partition_svg_keeps_original_content_and_adds_inner_witnesses():
    from generate_outer_partition_svg import build_svg, validate

    assert validate(build_svg()) == (25, 40, 13)


def test_inner_witness_geometry_is_c8_plus_five_isolated_points():
    from math import dist, isclose

    from generate_outer_partition_svg import inner_witness_points
    from outer_partition import LAMBDA

    points = inner_witness_points()
    center = points[0]
    vertices = points[1:9]
    added = points[9:]
    threshold = 2.0 * LAMBDA

    assert len(vertices) == 8
    assert len(added) == 4
    for k in range(8):
        p = vertices[k][1:3]
        q = vertices[(k + 1) % 8][1:3]
        assert isclose(dist(p, q), threshold, rel_tol=0.0, abs_tol=1e-12)

    isolated = [center, *added]
    for point in isolated:
        for other in points:
            if point is not other:
                assert dist(point[1:3], other[1:3]) > threshold + 1e-6
