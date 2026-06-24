import pytest
from numpy.testing import assert_array_equal

from engine import DataFrame, cat


@pytest.mark.xfail(reason="cat() is not finished yet", strict=True)
def test_cat_decode():
    df = DataFrame({"x": ["a", "b", "a"]})
    x_cat = cat(df.x)

    assert_array_equal(x_cat.decode(), ["a", "b", "a"])


@pytest.mark.xfail(reason="cat() is not finished yet", strict=True)
def test_cat_multi_col_decode():
    df = DataFrame({"x": ["a", "a", "b"], "y": [1, 2, 1]})
    xy_cat = cat(df.x, df.y)

    assert_array_equal(xy_cat.decode(), [("a", 1), ("a", 2), ("b", 1)])


@pytest.mark.xfail(reason="cat() is not finished yet", strict=True)
def test_cat_equality():
    df = DataFrame({"x": ["a", "b", "a"]})
    x_cat = cat(df.x)
    assert_array_equal(x_cat == "a", [True, False, True])


def test_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="same length"):
        cat(["a"], ["x", "y"])


@pytest.mark.xfail(reason="cat() is not finished yet", strict=True)
def test_cat_slice_preserves_categories():
    x_cat = cat(["a", "b", "a", "c"])
    sliced = x_cat[[0, 2]]

    assert_array_equal(sliced.decode(), ["a", "a"])
    assert_array_equal(sliced == "a", [True, True])
