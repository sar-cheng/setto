"""Expression trees used by lazy filters and future computed columns.

An expression can evaluate itself against a batch and report which source
columns it needs. The optimizer uses that column metadata for pushdown.

In this module, a "literal" means a constant value known before query
execution. It can be a scalar like `10`, or a nested value such as a list used
by a future `is_in()` expression.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

import numpy as np
import numpy.typing as npt

from engine._types import Batch

BinaryFn = Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]
UnaryFn = Callable[[npt.ArrayLike], npt.ArrayLike]

ColLike = np.ndarray[Any, Any] | list[Any]

NumericLiteral = int | float
TimeLiteral = date | time | datetime
NonNestedLiteral = NumericLiteral | TimeLiteral | str | bool | bytes
Literal = NonNestedLiteral | ColLike

ExprLike = ColLike | Literal


def _to_expr(x: ExprLike) -> Expr:
    """Convert user-provided values into expression nodes."""
    if isinstance(x, Expr):
        return x
    return Lit(x)


class Expr:
    """Base expression node.

    Every expression can evaluate against one batch and report the columns it
    depends on.
    """

    def evaluate(self, batch: Batch) -> npt.ArrayLike:
        """Evaluate this expression against a batch of columns."""
        raise NotImplementedError()

    def columns(self) -> set[str]:
        """Return source columns required to evaluate this expression."""
        raise NotImplementedError

    def __gt__(self, other: ExprLike) -> Expr:
        return BinaryOp(">", np.greater, self, _to_expr(other))

    def __lt__(self, other: ExprLike) -> Expr:
        return BinaryOp("<", np.less, self, _to_expr(other))

    def __eq__(self, other: ExprLike) -> Expr:
        return BinaryOp("==", np.equal, self, _to_expr(other))

    def __ne__(self, other: ExprLike) -> Expr:
        return BinaryOp("!=", np.not_equal, self, _to_expr(other))

    def __and__(self, other: ExprLike) -> Expr:
        return BinaryOp("&", np.logical_and, self, _to_expr(other))

    def __or__(self, other: ExprLike) -> Expr:
        return BinaryOp("|", np.logical_or, self, _to_expr(other))

    def __invert__(self) -> Expr:
        return UnaryOp("~", np.logical_not, self)

    def __bool__(self) -> None:
        raise TypeError("truth value of an Expr is ambiguous")

    # TODO: add ops like mul, add, rmul, radd?


class ColRef(Expr):
    """Reference to a column in a batch."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"col({self.name!r})"

    def _repr_html_(self):
        pass

    def evaluate(self, batch: Batch) -> np.ndarray:
        return batch[self.name]

    def columns(self) -> set[str]:
        return {self.name}


class Lit(Expr):
    """Constant value embedded in an expression tree."""

    def __init__(self, value: Literal):
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)

    def evaluate(self, _: Batch) -> Literal:
        return self.value

    def columns(self) -> set[str]:
        return set()


class BinaryOp(Expr):
    """Binary expression such as `col("price") > 300`."""

    def __init__(self, op: str, fn: BinaryFn, left: Expr, right: Expr):
        self.op = op
        self.fn = fn
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"

    def evaluate(self, batch: Batch) -> npt.ArrayLike:
        return self.fn(self.left.evaluate(batch), self.right.evaluate(batch))

    def columns(self) -> set[str]:
        return self.left.columns() | self.right.columns()


class UnaryOp(Expr):
    """Unary expression such as logical negation."""

    def __init__(self, op: str, fn: UnaryFn, operand: Expr):
        self.op = op
        self.fn = fn
        self.operand = operand

    def __repr__(self) -> str:
        return f"({self.op}{self.operand!r})"

    def evaluate(self, batch: Batch):
        return self.fn(self.operand.evaluate(batch))

    def columns(self) -> set[str]:
        return self.operand.columns()
