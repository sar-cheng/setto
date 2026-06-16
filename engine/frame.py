from __future__ import annotations

import html
import os

import numpy as np

from engine.expr import ColRef, Expr
from engine.op import Filter, Plan, Scan, Select


class DataFrame:
    _data: dict[str, np.array]

    def __init__(self, data: dict[str, np.array]):
        self._data = data

    def __getattr__(self, name: str):
        try:
            data = self._data
        except AttributeError:
            raise AttributeError("Data has not been initialized yet.")

        if name in data:
            return data[name]
        raise AttributeError(
            f"Column `{name}` does not exist in the dataframe!"
        )

    def __dir__(self):
        base_dir = super().__dir__()
        public_attrs = [k for k in base_dir if not k.startswith("_")]
        return public_attrs + list(self._data.keys())

    def _repr_html_(self):
        cols = list(self._data)
        nrows = len(self._data[cols[0]]) if cols else 0
        head = range(min(nrows, 10))

        # Make a single row with the column names
        th_elements = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
        thead = f"<thead><tr>{th_elements}</tr></thead>"

        # Loop through the first 10 rows and creates table data cells
        tr_elements = []
        for i in head:
            tds = "".join(
                f"<td>{html.escape(str(self._data[c][i]))}</td>" for c in cols
            )
            tr_elements.append(f"<tr>{tds}</tr>")
        tbody = f"<tbody>{''.join(tr_elements)}</tbody>"

        footer = f"<p>{nrows} rows × {len(cols)} columns</p>"

        th_names = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)

        # Grabs the type of the first element in each column (if rows exist)
        th_types = "".join(
            f"<th><em>{html.escape(type(self._data[c][0]).__name__ if nrows else '')}</em></th>"
            for c in cols
        )

        thead = (
            f"<thead>\n  <tr>{th_names}</tr>\n  <tr>{th_types}</tr>\n</thead>"
        )

        return f"<table>{thead}{tbody}</table>\n{footer}"

    def __repr__(self):
        cols = list(self._data)
        nrows = len(self._data[cols[0]]) if cols else 0
        head_count = min(nrows, 10)

        if not cols:
            return f"Empty table\n{nrows} rows × 0 columns"

        col_widths = {}
        for c in cols:
            max_w = len(str(c))
            for i in range(head_count):
                val_len = len(str(self._data[c][i]))
                if val_len > max_w:
                    max_w = val_len
            col_widths[c] = max_w

        header = "  ".join(str(c).ljust(col_widths[c]) for c in cols)

        rows = []
        for i in range(head_count):
            row_str = "  ".join(
                str(self._data[c][i]).ljust(col_widths[c]) for c in cols
            )
            rows.append(row_str)

        footer = f"\n{nrows} rows × {len(cols)} columns"

        output = [header] + rows

        if nrows > head_count:
            output.append("...")

        output.append(footer)

        return "\n".join(output)

    @property
    def shape(self) -> tuple[int, int]:
        pass


class LazyFrame:
    def __init__(self, plan: Plan):
        self._plan = plan

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(
                "Cols shouldn't start with '_'! I don't have it, or - rename it!"
            )
        if name in self._plan.schema:
            return ColRef(name)
        raise AttributeError(f"Column `{name} does not exist")

    def __dir__(self):
        base_dir = super().__dir__()
        public_attrs = [k for k in base_dir if not k.startswith("_")]
        return public_attrs + list(self._plan.schema)

    def filter(self, condition: Expr) -> LazyFrame:
        return LazyFrame(Filter(condition, self._plan))

    def select(self, *cols: ColRef) -> LazyFrame:
        return LazyFrame(Select([col.name for col in cols], self._plan))

    def collect(self) -> DataFrame:
        batches = list(self._plan.execute())
        return DataFrame(
            {
                col: np.concatenate([rows[col] for rows in batches])
                for col in batches[0]
            }
        )


def read(path: os.PathLike) -> DataFrame:
    return scan(path).collect()


def scan(path: os.PathLike) -> LazyFrame:
    return LazyFrame(Scan(path))
