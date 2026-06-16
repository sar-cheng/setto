import csv
import os
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from engine._types import Batch
from engine.expr import Expr

"""

df = l.scan("data.csv").filter(...)
aapl = df.filter(df.symbol == "aapl")      <-- new, df untouched

we want to make something that supports this:
df = scan("trades.parquet")
df.filter(df.price > 300).filter(df.qty < 5).select(df.symbol, df.price))

note i want to be able to display the col through df.price as well like how
pandas displays a col with df["price"], whilst maintaining the query plan / expression tree (? not sure if im supposed to do this)

for a table / df, what operations had to be fast?
- scanning a lot of rows, usually filter
- agg by col

- groupby(x) then agg(y)
- join
- dedup / unique

- sort / order by (e.g. order by x desc)
- top n (e.g. order by ... limit 10)
- window functions - running totals, moving averages, rank within a partition (e.g. over (partition by))

df = mapping of col name -> col array
"""

"""
data flows through operators in chunks - smaller versions of a df - a dict of arrays, where each is a slice

groupbyop
sum, mean, median

NOTE:
- scan is at the bottom, other ops stack on top of it
- each op takes batches of data from the ops below, and passes them up

We have PLAN nodes, and EXPR nodes
- we do .scan(),filter().select() -> in memory: Select( Filter( Scan() ) )
- do .collect() to convert the plan to the IR, i.e. change from Russian doll to list of pointers
- optimizer runs is rules

"""


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
