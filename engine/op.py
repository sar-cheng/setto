import csv
import os
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from engine._types import Batch
from engine.expr import Expr


class Plan:
    schema: Sequence[str]

    def execute(self) -> Iterator[Batch]:
        pass


class Scan(Plan):
    def __init__(
        self, path: str | os.PathLike[str], *, batch_size: int = 10_000
    ):
        self.path, self.batch_size = path, batch_size

        with open(self.path) as f:
            reader = csv.DictReader(f)
            self.schema = reader.fieldnames
            if not self.schema:
                raise ValueError("CSV file is empty")

    def _to_array(self, vals: list[str]) -> np.ndarray:
        try:
            return np.array(vals, dtype=float)
        except ValueError:
            return np.array(vals)

    def execute(self) -> Iterator[Batch]:
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


class Filter(Plan):
    def __init__(self, predicate: Expr, child: Plan):
        """
        predicate: the condition to filter on, an expression tree / node
        """
        self.predicate = predicate
        self.child = child
        self.schema = child.schema

    def execute(self) -> Iterator[Batch]:
        for batch in self.child.execute():
            mask: npt.ArrayLike[bool] = self.predicate.evaluate(batch)
            yield {
                col_name: col_vals[mask] for col_name, col_vals in batch.items()
            }


class Select(Plan):
    def __init__(self, cols: list[str], child: Plan):
        self.cols = cols
        self.child = child
        self.schema = child.schema

    def execute(self) -> Iterator[Batch]:
        for batch in self.child.execute():
            yield {col: batch[col] for col in self.cols}
