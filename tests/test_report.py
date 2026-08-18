from afterimage.bench.harness import aggregate
from afterimage.bench.report import format_table


def test_format_table_marks_unstable_metric():
    results = {
        "A": aggregate([{"gb": v} for v in [1, 1, 1, 1]]),
        "B": aggregate([{"gb": v} for v in [1, 1, 1, 100]]),
    }
    table = format_table(results)
    assert "A" in table
    assert "B" in table
    assert "*" in table


def test_format_table_handles_empty_results():
    assert format_table({}) == "(no results)"
