"""Shared pytest fixtures for small CSV-backed examples."""

import pytest


@pytest.fixture
def sample_rows():
    """Return a compact trade dataset used across tests."""
    return [
        {"symbol": "AAPL", "price": 205.5, "qty": 1},
        {"symbol": "MSFT", "price": 417.5, "qty": 4},
    ]


@pytest.fixture
def write_csv(tmp_path, sample_rows):
    """Return a helper that writes rows to a temporary CSV file."""
    def _write(rows=None):
        rows = sample_rows if rows is None else rows

        path = tmp_path / "input.csv"
        header = rows[0].keys()

        with path.open("w") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(",".join(str(row[col]) for col in header) + "\n")
        return path

    return _write


@pytest.fixture
def csv_path(write_csv, sample_rows):
    """Return the default sample rows written as a temporary CSV."""
    return write_csv(sample_rows)
