from __future__ import annotations

import html

import numpy as np
import numpy.typing as npt


class CatArray:
    _codes: npt.NDArray[np.int64] | None
    _categories: npt.NDArray[np.object_] | None

    def __init__(self, codes: npt.ArrayLike, categories: npt.ArrayLike):
        self._codes = np.asarray(codes, dtype=np.int64)
        self._categories = categories

    def __len__(self):
        return len(self._codes)

    def __getitem__(
        self, key: int | slice | npt.NDArray[np.integer] | list[int]
    ) -> npt.str_ | CatArray:
        sub = self._codes[key]
        if np.ndim(sub) == 0:
            return self._categories[sub]
        return CatArray(sub, self._categories)

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> npt.ArrayLike:
        return self._codes if dtype is None else self._codes.astype(dtype)

    def decode(self) -> npt.NDArray[np.object_]:
        return self._categories[self._codes]

    def __repr__(self) -> str:
        return repr(self.decode())

    def _repr_html_(self) -> str:
        values = np.atleast_1d(self.decode())
        return "<br>".join(html.escape(str(v)) for v in values)
