import sympy as sp

from verify_qrho_counterexample import H, RHO0, counterexample_centers, verify


def test_exact_countercertificate_covers_the_entire_tail_interval():
    report = verify()
    assert report["status"] == "exact_countercertificate_verified"
    assert report["limit_geometric_points"] == 48
    assert report["owner_tagged_incidences"] == 128
    assert report["fixed_cover_centers"] == 18
    assert report["covered_limit_points"] == 48
    assert len(counterexample_centers()) == 18
    assert sp.simplify(H - sp.cos(sp.pi / 8)) == 0
    assert 0 < RHO0 < 1
    assert 0.961 < float(RHO0) < 0.962
