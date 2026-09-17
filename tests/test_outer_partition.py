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


def test_outer_partition_svg_has_25_circles_and_40_points():
    from generate_outer_partition_svg import build_svg, validate

    assert validate(build_svg()) == (25, 40)
