"""Load asset returns from a local CSV file.

Expected layout: a date column first, then one numeric column per asset.

    date,AAPL,MSFT
    2024-01-02,101.5,372.1
    2024-01-03,100.9,370.4

Pass `kind="prices"` for price levels, which are converted to simple returns, or
`kind="returns"` when the file already holds period returns as decimals (0.01 for one percent).

Deliberately local. Downloading market data would add a dependency and a network call, and the
downloaded file belongs in version control decisions rather than inside a solver run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple period returns from price levels. Drops the first row, which has no prior price."""
    if (prices <= 0.0).to_numpy().any():
        raise ValueError("prices must be strictly positive")
    return prices.pct_change().iloc[1:]


def load_returns_csv(
    path: str, kind: str = "returns", date_column: str | None = None
) -> pd.DataFrame:
    """Read a CSV of returns or prices and return a validated returns frame.

    Raises ValueError on empty files, non-numeric columns, missing values, duplicate or
    unsorted dates, and non-positive prices.
    """
    if kind not in ("returns", "prices"):
        raise ValueError("kind must be 'returns' or 'prices'")

    frame = pd.read_csv(path)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("CSV needs a date column and at least one asset column")

    index_column = date_column if date_column is not None else frame.columns[0]
    if index_column not in frame.columns:
        raise ValueError(f"date column {index_column!r} is not in the file")

    frame = frame.set_index(index_column)
    frame.index = pd.to_datetime(frame.index, errors="raise")
    frame.index.name = "date"

    non_numeric = [
        column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(f"non-numeric asset columns: {non_numeric}")
    if frame.isna().to_numpy().any():
        raise ValueError("missing values are not allowed; clean or drop them first")
    if not frame.index.is_unique:
        raise ValueError("duplicate dates in the file")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("dates must be sorted oldest first")

    returns = prices_to_returns(frame) if kind == "prices" else frame.astype(float)
    if len(returns) < 2:
        raise ValueError("need at least two periods of returns")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns contain infinite values")
    return returns
