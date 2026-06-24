"""Public functions for building frames and arrays."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy.typing as npt

from .frame import DataFrame, LazyFrame
from .plan import Scan

if TYPE_CHECKING:
    from .cat import CatArray


def read(path: os.PathLike) -> DataFrame:
    """Read a CSV file eagerly by scanning and collecting it."""
    return scan(path).collect()


def scan(path: os.PathLike) -> LazyFrame:
    """Create a lazy CSV scan without reading row data yet."""
    return LazyFrame(Scan(path))


def cat(*cols: npt.ArrayLike) -> CatArray:
    """Encode columns as a categorical array.

    This is a light public wrapper. The pandas-backed implementation is imported
    only when `cat()` is called, so plain `import engine` stays cheap.
    """
    from .cat import cat as _cat

    return _cat(*cols)
