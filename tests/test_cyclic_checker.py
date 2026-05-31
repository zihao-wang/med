from med.cyclic_polytope.checker import CyclicPolytopeChecker


def test_checker_reports_query_coverage():
    checker = CyclicPolytopeChecker(m=4, k=2)
    result = checker.check(d=4)

    assert result.feasible
    assert result.details["checks"] == 10
    assert result.details["total_queries"] == 10
    assert result.details["checked_fraction"] == 1.0
    assert result.details["checks_by_size"] == {1: 4, 2: 6}
