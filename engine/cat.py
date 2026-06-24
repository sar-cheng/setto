"""Categorical arrays.

`CatArray` stores integer codes plus a category lookup table. Multi-column
categoricals are encoded by factorizing each column, packing row keys with
mixed-radix arithmetic, then compressing the observed keys back to dense codes.
"""

from __future__ import annotations

import html
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

_MAX_DENSE_SPACE = 1_000_000


def cat(*cols: npt.ArrayLike) -> CatArray:
    """Encode one or more same-length columns as a categorical array.

    The returned `CatArray` stores integer codes plus one value per category.
    With one column, categories are the original values. With multiple columns,
    each category is the row tuple across those columns.

    Missing values use code -1. If any input column is missing a row, that
    row is missing in the result, and `decode()` returns None for it.
    """
    if not cols:
        raise ValueError("cat needs at least one column")

    arrays: list[npt.NDArray[Any]] = [np.asarray(c) for c in cols]
    n = len(arrays[0])
    if any(len(c) != n for c in arrays):
        raise ValueError("all columns must have the same length")

    # Factorize columns separately so the row path can stay vectorized.
    per_codes: list[npt.NDArray[np.int64]] = []
    per_uniques: list[npt.NDArray[np.object_]] = []
    sizes: list[int] = []
    missing = np.zeros(n, dtype=bool)

    for col in arrays:
        codes, uniques = pd.factorize(col, use_na_sentinel=True)
        codes = codes.astype(np.int64, copy=False)

        missing |= codes == -1
        per_codes.append(codes)
        per_uniques.append(_object_array(uniques))
        sizes.append(len(uniques))

    final_codes = np.full(n, -1, dtype=np.int64)
    valid = ~missing
    if not valid.any():
        return CatArray(final_codes, _object_array([]))

    space = math.prod(sizes)
    if space < 2**63:
        combined = np.zeros(valid.sum(), dtype=np.int64)
        for codes, size in zip(per_codes, sizes):
            combined = combined * size + codes[valid]

        valid_codes, keys = _factorize_combined(combined, space)
        categories = _categories_from_keys(keys, per_uniques, sizes)
    else:
        values = _row_tuples(arrays, valid)
        valid_codes, categories = pd.factorize(values, use_na_sentinel=False)
        categories = _object_array(categories)

    final_codes[valid] = valid_codes.astype(np.int64, copy=False)
    return CatArray(final_codes, categories)


def _factorize_combined(
    combined: npt.NDArray[np.int64], space: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Compress packed row keys to dense codes."""
    if space <= _MAX_DENSE_SPACE and space <= 4 * len(combined):
        # Dense remapping is fast, but only safe while the key space is small.
        seen = np.zeros(space, dtype=bool)
        seen[combined] = True

        keys = np.flatnonzero(seen).astype(np.int64, copy=False)
        remap = np.empty(space, dtype=np.int64)
        remap[keys] = np.arange(len(keys), dtype=np.int64)
        return remap[combined], keys

    codes, keys = pd.factorize(combined, use_na_sentinel=False)
    return (
        codes.astype(np.int64, copy=False),
        keys.astype(np.int64, copy=False),
    )


def _categories_from_keys(
    keys: npt.NDArray[np.int64],
    per_uniques: list[npt.NDArray[np.object_]],
    sizes: list[int],
) -> npt.NDArray[np.object_]:
    """Decode packed row keys back into scalar or tuple categories."""
    if len(per_uniques) == 1:
        return _object_array(per_uniques[0][keys])

    values: list[tuple[Any, ...]] = []
    for raw_key in keys:
        key = int(raw_key)
        component_codes: list[int] = [0] * len(sizes)

        for i in range(len(sizes) - 1, -1, -1):
            component_codes[i] = key % sizes[i]
            key //= sizes[i]

        values.append(
            tuple(per_uniques[i][component_codes[i]] for i in range(len(sizes)))
        )

    return _object_array(values)


def _row_tuples(
    arrays: list[npt.NDArray[Any]], valid: npt.NDArray[np.bool_]
) -> npt.NDArray[np.object_]:
    """Fallback for packed key spaces that would overflow int64."""
    return _object_array(
        tuple(array[i] for array in arrays) for i in np.flatnonzero(valid)
    )


def _object_array(values: Iterable[Any]) -> npt.NDArray[np.object_]:
    """Build a 1D object array, even when values are tuples."""
    values = list(values)
    out = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        out[i] = value
    return out


class CatArray:
    """A categorical array of integer codes.

    Codes index into `_categories`; code -1 means missing. Multi-column
    categories are stored as tuples.
    """

    _codes: npt.NDArray[np.int64]
    _categories: npt.NDArray[np.object_]

    def __init__(self, codes: npt.ArrayLike, categories: npt.ArrayLike) -> None:
        self._codes = np.asarray(codes, dtype=np.int64)
        self._categories = np.asarray(categories, dtype=object)

    def __len__(self) -> int:
        return len(self._codes)

    def decode(self) -> npt.NDArray[np.object_]:
        """Reconstruct values; missing rows become None."""
        out = np.empty(len(self._codes), dtype=object)

        valid = self._codes >= 0
        out[valid] = self._categories[self._codes[valid]]
        out[~valid] = None
        return out

    def __repr__(self) -> str:
        return repr(self.decode())

    def _repr_html_(self) -> str:
        values = np.atleast_1d(self.decode())
        return "<br>".join(html.escape(str(v)) for v in values)

    def __getitem__(
        self, key: int | slice | npt.NDArray[Any] | list[int]
    ) -> Any | CatArray:
        """Return a decoded scalar or a sliced categorical array."""
        sub = self._codes[key]
        if np.ndim(sub) == 0:
            return None if sub < 0 else self._categories[sub]
        return CatArray(sub, self._categories)

    def take(self, indices: npt.ArrayLike) -> CatArray:
        """Return rows at integer positions while preserving categories."""
        indices = np.asarray(indices, dtype=np.int64)
        return CatArray(self._codes[indices], self._categories)

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[Any]:
        """Expose raw codes to NumPy."""
        return self._codes if dtype is None else self._codes.astype(dtype)

    def isna(self) -> npt.NDArray[np.bool_]:
        """Return a mask of rows encoded as missing."""
        return self._codes < 0

    def __eq__(self, x: Any) -> npt.NDArray[np.bool_]:
        # Equality has to work for both scalar and tuple categories.
        matches = np.flatnonzero([c == x for c in self._categories])
        if len(matches) == 0:
            return np.zeros(len(self._codes), dtype=bool)

        return self._codes == matches[0]

    __hash__ = None
