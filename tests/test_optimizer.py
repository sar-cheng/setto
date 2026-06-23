import pytest

from engine import DataFrame, scan
from tests.utils import assert_df_equal, plan_shape


def test_projection_pushdown_preserves_result(csv_path):
    lf = scan(csv_path)
    actual = lf.filter(lf.price > 300).select(lf.symbol).collect()

    assert_df_equal(actual, DataFrame({"symbol": ["MSFT"]}))


def test_projection_pushdown_limits_scan_columns(csv_path):
    lf = scan(csv_path)
    q = lf.filter(lf.price > 300).select(lf.symbol)

    assert plan_shape(q.optimized_plan()) == (
        "select",
        ("symbol",),
        (
            "filter",
            frozenset({"price"}),
            (
                "scan",
                ("symbol", "price"),
            ),
        ),
    )


def test_predicate_pushdown_reorders_filter_below_select(csv_path):
    lf = scan(csv_path)
    q = lf.select(lf.symbol).filter(lf.symbol == "AAPL")

    assert plan_shape(q.optimized_plan()) == (
        "select",
        ("symbol",),
        (
            "filter",
            frozenset({"symbol"}),
            ("scan", ("symbol",)),
        ),
    )


def test_invalid_filter_select_raises_for_missing_filter_column(csv_path):
    lf = scan(csv_path)
    selected = lf.select(lf.symbol)

    with pytest.raises(ValueError, match="price"):
        selected.filter(lf.price > 300)
