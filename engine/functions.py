import math
import os

import numpy as np
import numpy.typing as npt
import pandas as pd

from .cat import CatArray
from .frame import DataFrame, LazyFrame
from .plan import Scan


def read(path: os.PathLike) -> DataFrame:
    return scan(path).collect()


def scan(path: os.PathLike) -> LazyFrame:
    return LazyFrame(Scan(path))


def cat(*cols: npt.ArrayLike) -> CatArray:
    if not cols:
        raise ValueError("cat needs at least one column")

    arrays = [np.asarray(c) for c in cols]
    n = len(arrays[0])
    if any(len(c) != n for c in arrays):
        raise ValueError("all columns must have the same length")

    # hash-factorize each col
    per_codes: list[npt.NDArray[np.int64]] = []
    per_uniques: list[npt.NDArray[np.object_]] = []
    sizes: list[int] = []
    for col in cols:
        # NOTE: Nan becomes a real category, so the integer combine below
        # never sees a -1 that'd just wrap around
        c, u = pd.factorize(col, use_na_sentinel=False)
        per_codes.append(c.astype(np.int64))
        per_uniques.append(np.asarray(u, dtype=object))
        sizes.append(len(u))

    space = math.prod(sizes)
    if space < 2**63:
        # mixed-radix combine
        combined = np.zeros(len(cols[0]), dtype=np.int64)
        for c, s in zip(per_codes, sizes):
            # if A has 3 categories and B has 4, then
            # combined = a_code * 4 + b_code
            combined = combined * s + c

        # compress to dense codes
        if space < 2**31 and space <= 4 * len(combined):
            # assign ints to only the keys that were actually seen
            seen = np.zeros(space, dtype=bool)
            seen[combined] = True

            remap = np.empty(space, np.int64)
            remap[seen] = np.arange(seen.sum())

            final_codes = remap[combined]
        else:
            # large / high-cardinality
            final_codes, _ = pd.factorize(combined)
            return final_codes
    # else:
    # factorise row tuples directly (slower)

    # return CatArray(final_codes.astype(np.int64), categories)
    # TODO: finish this
