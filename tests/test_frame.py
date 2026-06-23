from engine import DataFrame, read, scan
from tests.utils import assert_df_equal


def test_read_collects_csv(csv_path):
    df = read(csv_path)

    expected = DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "price": [205.5, 417.5],
            "qty": [1, 4],
        }
    )
    assert_df_equal(df, expected)


def test_filter_select_collect(csv_path):
    lf = scan(csv_path)
    actual = lf.filter(lf.price > 300).select(lf.symbol).collect()

    expected = DataFrame({"symbol": ["MSFT"]})
    assert_df_equal(actual, expected)


def test_empty_filter_preserves_schema(write_csv):
    path = write_csv(
        [
            {"symbol": "AAPL", "price": 205.5, "qty": 1},
        ]
    )
    lf = scan(path)
    out = lf.filter(lf.price > 999).select(lf.symbol).collect()

    expected = DataFrame({"symbol": []})
    assert_df_equal(out, expected)


def test_df_shape(csv_path):
    df = read(csv_path)
    assert df.shape == (2, 3)


def test_empty_df_shape():
    df = DataFrame({})
    assert df.shape == (0, 0)


def test_df_no_rows_preserves_column_names(csv_path):
    lf = scan(csv_path)
    out = lf.filter(lf.price > 999).select(lf.symbol).collect()

    assert_df_equal(out, DataFrame({"symbol": []}))
    assert out.shape == (0, 1)
