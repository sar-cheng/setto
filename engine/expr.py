from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

import numpy as np
import numpy.typing as npt

from engine._types import Batch

"""
TODO: consider match instead of OOP?

Expression tree:
- leaves = cols and literals
- turn columns into a new col / mask

Logical plan (the query plan)
- Scan -> Filter(pred) -> Filter(pred) -> Select(cols)
- expressions live INSIDE plan nodes
- a Filter node holds an expression tree (the predicate),
    a Select node holds a list of expression trees (the output cols)
"""

BinaryFn = Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]
UnaryFn = Callable[[npt.ArrayLike], npt.ArrayLike]

ColLike = np.ndarray[Any, Any] | list[Any]

"""
Literals

NOTE: the meaning of "literal" here is not just "a simple number or string".
It means a constant / hard-coded value provided by the user that is known
before the query runs.

Examples of nested literals in expressions:
- df.department.is_in(['HR', 'Engineering', 'Sales'])
- df.price * np.array([1.1, 1.2, ...])

"""
NumericLiteral = int | float
TimeLiteral = date | time | datetime
NonNestedLiteral = NumericLiteral | TimeLiteral | str | bool | bytes
Literal = NonNestedLiteral | ColLike

ExprLike = ColLike | Literal


def col(name: str) -> ColRef:
    return ColRef(name)


def _to_expr(x: ExprLike) -> Expr:
    if isinstance(x, Expr):
        return x
    return Lit(x)


class Expr:
    """
    df.col > 100
    Every Expr node knows how to turn a dict[str, array] (one batch, or the
    whole frame) into an array.
    """

    def evaluate(self, batch: Batch) -> npt.ArrayLike:
        raise NotImplementedError("poop")

    def __gt__(self, other: ExprLike) -> Expr:
        return BinaryOp(np.greater, self, _to_expr(other))

    def __lt__(self, other: ExprLike) -> Expr:
        return BinaryOp(np.less, self, _to_expr(other))

    def __eq__(self, other: ExprLike) -> Expr:
        # NOTE: by defining this method, we've made Expr unhashable
        # Py will set __hash = None
        return BinaryOp(np.equal, self, _to_expr(other))

    def __ne__(self, other: ExprLike) -> Expr:
        return BinaryOp(np.not_equal, self, _to_expr(other))

    def __bool__(self) -> None:
        raise TypeError("truth value of an Expr is ambiguous")

    # Bitwise
    def __and__(self, other: ExprLike) -> Expr:
        return BinaryOp(np.logical_and, self, _to_expr(other))

    def __or__(self, other):
        return BinaryOp(np.logical_or, self, _to_expr(other))

    def __invert__(self):
        return UnaryOp(np.logical_not, self)

    # TODO: add ops like mul, add, rmul, radd?


class ColRef(Expr):
    def __init__(self, name: str):
        self.name = name

    def _repr_html_(self):
        pass

    def evaluate(self, batch: Batch) -> np.ndarray:
        return batch[self.name]


class Lit(Expr):
    def __init__(self, value: Literal):
        self.value = value

    def evaluate(self, _: Batch) -> Literal:
        return self.value


class BinaryOp(Expr):
    def __init__(self, fn: BinaryFn, left: Expr, right: Expr):
        self.fn = fn
        self.left = left
        self.right = right

    def evaluate(self, batch: Batch) -> npt.ArrayLike:
        return self.fn(self.left.evaluate(batch), self.right.evaluate(batch))


class UnaryOp(Expr):
    def __init__(self, fn: UnaryFn, operand: Expr):
        self.fn, self.operand = fn, operand

    def evaluate(self, batch: Batch):
        return self.fn(self.operand.evaluate(batch))
