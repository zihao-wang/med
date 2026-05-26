from med.cyclic_polytope.experiment import _make_check_dimension


def test_cyclic_check_dimension_reports_feasibility():
    check_dimension = _make_check_dimension(m=4, k=2)
    result = check_dimension(4)

    assert result is True
