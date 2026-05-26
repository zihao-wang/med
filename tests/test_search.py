"""Tests for search.py — binary search over dimensions."""


class ThresholdChecker:
    """Passes when d >= threshold."""

    def __init__(self, threshold: int):
        self.threshold = threshold

    def __call__(self, d: int) -> bool:
        return d >= self.threshold


def test_binary_search_always_fail():
    from med.search import binary_search_med

    result = binary_search_med(lambda _d: False, left0=0, max_range=10, verbose=False)
    assert result["med"] is None
    assert len(result["search_path"]) > 0


def test_binary_search_always_pass():
    from med.search import binary_search_med

    result = binary_search_med(lambda _d: True, left0=0, max_range=10, verbose=False)
    assert result["med"] == 1  # smallest checked


def test_binary_search_threshold():
    from med.search import binary_search_med

    result = binary_search_med(ThresholdChecker(5), left0=0, max_range=20, verbose=False)
    assert result["med"] == 5


def test_binary_search_threshold_warm_start():
    from med.search import binary_search_med

    # Warm start from 3 means search is [4, 23]
    result = binary_search_med(ThresholdChecker(5), left0=3, max_range=20, verbose=False)
    assert result["med"] == 5


def test_binary_search_threshold_at_upper_bound():
    from med.search import binary_search_med

    result = binary_search_med(ThresholdChecker(10), left0=0, max_range=10, verbose=False)
    assert result["med"] == 10


def test_binary_search_threshold_above_upper_bound():
    from med.search import binary_search_med

    result = binary_search_med(ThresholdChecker(50), left0=0, max_range=10, verbose=False)
    assert result["med"] is None


def test_search_path_entries():
    from med.search import binary_search_med

    result = binary_search_med(ThresholdChecker(3), left0=0, max_range=5, verbose=False)
    assert len(result["search_path"]) >= 1
    for entry in result["search_path"]:
        assert "dimension" in entry
        assert "feasible" in entry
        assert "time" in entry
