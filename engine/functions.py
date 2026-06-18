import os

from .frame import DataFrame, LazyFrame
from .op import Scan


def read(path: os.PathLike) -> DataFrame:
    return scan(path).collect()


def scan(path: os.PathLike) -> LazyFrame:
    return LazyFrame(Scan(path))


# def cat(*cols: npt.NDArray) -> CatArray:
#     # hash-factorise each col
#     pass
