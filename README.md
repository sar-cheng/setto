# setto

This is a tiny dataframe/query-engine project for me to learn how lazy
execution and simple query optimization work whilst brushing up on programming.

It's similar to and heavily inspired by Polars and Pandas. I wanted to make the
internals small enough to read, poke at, and reason about.

## What It Supports

- CSV `scan()` and eager `read()`
- lazy `filter()`, `select()`, and `collect()`
- expression trees such as `lf.price > 300`
- `explain()` for inspecting plans
- projection pushdown
- simple predicate pushdown
- small `DataFrame` display helpers
- experimental categorical arrays via `cat()`

Eager reads are intentionally thin:

```python
read(path) == scan(path).collect()
```

## Example

```python
from engine import scan

lf = scan("data/trades.csv")

q = (
    lf
    .filter(lf.price > 300)
    .select(lf.symbol)
)

print(q.explain())
print(q.explain(optimized=True))
print(q.collect())
```

The optimized plan only scans columns needed by the final output and predicate.

## Development

Install the small development dependency set:

```bash
python -m pip install numpy pandas pytest ruff
```

Run tests:

```bash
python -m pytest tests
```

Run formatting/linting:

```bash
ruff format .
ruff check .
```
