"""Plan nodes for lazy query execution.

Plans form a tree: user calls from a LazyFrame build `Scan`, `Filter`, and
`Select` nodes, then the optimizer rewrites the tree before execution.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from engine._types import Batch
from engine.expr import Expr


class Plan:
    """Base interface for query plan nodes."""

    schema: Sequence[str]

    def execute(self) -> Iterator[Batch]:
        """Yield batches of column arrays."""
        raise NotImplementedError

    def explain(self, indent: int = 0) -> str:
        """Render this node and its children as an indented tree."""
        raise NotImplementedError


class Scan(Plan):
    """Read selected columns from a CSV file in batches."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        columns: list[str] | None = None,
        batch_size: int = 10_000,
    ):
        self.path, self.batch_size = path, batch_size

        with open(self.path) as f:
            reader = csv.DictReader(f)
            self.file_schema = reader.fieldnames
            if not self.file_schema:
                raise ValueError("CSV file is empty")

            self.schema = columns if columns is not None else self.file_schema

    def _to_array(self, vals: list[str]) -> np.ndarray:
        try:
            return np.array(vals, dtype=float)
        except ValueError:
            return np.array(vals)

    def execute(self) -> Iterator[Batch]:
        """Stream CSV rows into columnar NumPy batches."""
        with open(self.path) as f:
            reader = csv.DictReader(f)

            buf: list[dict[str, str]] = []
            for row in reader:
                buf.append(row)
                if len(buf) == self.batch_size:
                    yield {
                        col: self._to_array([row[col] for row in buf])
                        for col in self.schema
                    }
                    buf = []

            if buf:
                yield {
                    col: self._to_array([row[col] for row in buf])
                    for col in self.schema
                }

    def explain(self, indent: int = 0) -> str:
        pad = " " * indent
        return f"{pad}Scan(path={self.path}, schema={list(self.schema)})"


class Filter(Plan):
    """Keep rows whose predicate evaluates to true."""

    def __init__(self, predicate: Expr, child: Plan):
        self.predicate = predicate
        self.child = child
        self.schema = child.schema

    def execute(self) -> Iterator[Batch]:
        """Apply the predicate mask to each child batch."""
        for batch in self.child.execute():
            mask: npt.ArrayLike[bool] = self.predicate.evaluate(batch)
            yield {
                col_name: col_vals[mask] for col_name, col_vals in batch.items()
            }

    def explain(self, indent: int = 0) -> str:
        pad = " " * indent
        return (
            f"{pad}Filter(predicate={self.predicate})\n"
            f"{self.child.explain(indent + 2)}"
        )


class Select(Plan):
    """Keep only a subset of columns from the child plan."""

    def __init__(self, cols: list[str], child: Plan):
        self.cols = cols
        self.child = child
        self.schema = cols

    def execute(self) -> Iterator[Batch]:
        """Project each child batch down to selected columns."""
        for batch in self.child.execute():
            yield {col: batch[col] for col in self.cols}

    def explain(self, indent: int = 0) -> str:
        pad = " " * indent
        return (
            f"{pad}Select(cols={self.cols})\n{self.child.explain(indent + 2)}"
        )
