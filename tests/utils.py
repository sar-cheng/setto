"""Test helpers for comparing frames and inspecting plan structure."""

import numpy as np

from engine import DataFrame
from engine.plan import Filter, Plan, Scan, Select


def assert_df_equal(
    actual: DataFrame, expected: DataFrame, check_dtype=False
) -> None:
    assert list(actual._data) == list(expected._data)

    for col in expected._data:
        np.testing.assert_array_equal(actual._data[col], expected._data[col])
        if check_dtype:
            assert actual._data[col].dtype == expected._data[col].dtype


def plan_shape(plan: Plan):
    """Return a stable nested tuple representation of a plan tree."""
    match plan:
        case Scan():
            return ("scan", tuple(plan.schema))
        case Select():
            return ("select", tuple(plan.cols), plan_shape(plan.child))
        case Filter():
            return (
                "filter",
                frozenset(plan.predicate.columns()),
                plan_shape(plan.child),
            )

    raise AssertionError(f"Unknown plan node: {type(plan).__name__}")
