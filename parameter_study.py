"""Run the small real-data parameter sensitivity study."""

from __future__ import annotations

import argparse

import pandas as pd

from backtest import STRATEGIES, compare, compare_costs, run_backtest, summarize
from data import load_returns_csv

DEFAULT_LOOKBACK = 60
DEFAULT_COST_BPS = 10.0
DEFAULT_REBALANCE_EVERY = 1
DEFAULT_FORECAST_METHOD = "ewma"


def sensitivity_table(
    returns: pd.DataFrame,
    strategy: str,
    parameter: str,
    values: tuple[object, ...],
    **baseline,
) -> pd.DataFrame:
    """Vary one backtest argument and return summary metrics for each value."""
    allowed = {"cost_bps", "lookback", "rebalance_every", "forecast_method"}
    if parameter not in allowed:
        raise ValueError(f"parameter must be one of {sorted(allowed)}")
    if not values:
        raise ValueError("values cannot be empty")

    rows = []
    for value in values:
        options = baseline.copy()
        options[parameter] = value
        metrics = summarize(run_backtest(returns, strategy, **options))
        rows.append({"parameter": parameter, "value": value, **metrics.to_dict()})
    return pd.DataFrame(rows).set_index(["parameter", "value"])


def _print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.T.to_string(float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-data portfolio parameter comparisons")
    parser.add_argument("--csv", default="data/market_prices.csv")
    parser.add_argument("--kind", choices=("returns", "prices"), default="prices")
    parser.add_argument("--periods", type=int, default=150)
    parser.add_argument(
        "--sensitivity-strategy", choices=STRATEGIES, default="multi_period"
    )
    args = parser.parse_args()

    returns = load_returns_csv(args.csv, kind=args.kind)
    baseline = {
        "lookback": DEFAULT_LOOKBACK,
        "periods": args.periods,
        "cost_bps": DEFAULT_COST_BPS,
        "rebalance_every": DEFAULT_REBALANCE_EVERY,
        "forecast_method": DEFAULT_FORECAST_METHOD,
    }

    _print_table("baseline comparison", compare(returns, **baseline))
    _print_table(
        "cost sensitivity",
        compare_costs(returns, cost_levels=(5.0, 10.0, 20.0), **{k: v for k, v in baseline.items() if k != "cost_bps"}),
    )
    for parameter, values in (
        ("lookback", (20, 60, 120)),
        ("rebalance_every", (1, 5, 21)),
        ("forecast_method", ("ewma", "lag1", "ridge")),
    ):
        _print_table(
            f"{args.sensitivity_strategy} sensitivity: {parameter}",
            sensitivity_table(returns, args.sensitivity_strategy, parameter, values, **baseline),
        )


if __name__ == "__main__":
    main()
