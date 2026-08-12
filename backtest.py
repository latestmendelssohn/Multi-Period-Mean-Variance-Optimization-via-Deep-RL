"""Walk-forward backtest for the multi-period optimizer and its benchmarks.

Every strategy sees only returns strictly before the period it trades into, holds a long-only
fully invested portfolio, and pays a linear cost on traded notional. All strategies start from
an equal-weight position, which is not charged, so the comparison is not distorted by an
arbitrary initial trade.

By default the L1 penalty inside the objective equals the cost actually charged, so the optimizer
is not tuned against a cost it never pays. Pass `lambda_` to override that.

Run `python backtest.py` for the comparison table on synthetic data.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data import load_returns_csv
from multi_period_admm import (
    GAMMA,
    LOOKBACK,
    estimate_window,
    generate_synthetic_market_data,
    lambda_from_cost,
    solve_target_weights,
)
from rl_policy import predict_weights, train_policy

COST_BPS = 10.0
HORIZON = 5
PERIODS_PER_YEAR = 252
STRATEGIES = (
    "buy_and_hold",
    "equal_weight",
    "minimum_variance",
    "single_period",
    "multi_period",
    "rl_policy",
)


def run_backtest(
    returns: pd.DataFrame,
    strategy: str,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    gamma: float = GAMMA,
    lambda_: float | None = None,
    cost_bps: float = COST_BPS,
    periods: int | None = None,
) -> pd.DataFrame:
    """Run one strategy and return per-period returns, turnover, and cost.

    `lambda_` defaults to the charged cost. Deterministic strategies estimate parameters from
    the trailing window before each tested period. `rl_policy` trains once on all returns before
    the first tested period, then acts deterministically without seeing test returns.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if lookback < 2 or lookback >= len(returns):
        raise ValueError("lookback must be at least 2 and smaller than the number of rows")
    if cost_bps < 0.0:
        raise ValueError("cost_bps cannot be negative")

    n_assets = returns.shape[1]
    first = lookback
    if periods is not None:
        if periods < 1:
            raise ValueError("periods must be at least 1")
        first = max(lookback, len(returns) - periods)

    penalty = lambda_from_cost(cost_bps) if lambda_ is None else lambda_
    if penalty < 0.0:
        raise ValueError("lambda_ cannot be negative")
    cost_rate = cost_bps / 10_000.0
    equal_weights = np.full(n_assets, 1.0 / n_assets)

    trained_policy = None
    if strategy == "rl_policy":
        training = returns.iloc[:first]
        rl_lookback = min(20, max(2, len(training) // 3))
        if len(training) <= rl_lookback:
            raise ValueError("rl_policy needs more pre-test history")
        trained_policy = train_policy(
            training,
            equal_weights,
            lookback=rl_lookback,
            cost_bps=cost_bps,
        )

    held = equal_weights
    records = []

    for index in range(first, len(returns)):
        history = returns.iloc[index - lookback : index]
        realized = returns.iloc[index].to_numpy()

        if strategy == "buy_and_hold":
            target = held
        elif strategy == "equal_weight":
            target = equal_weights
        else:
            mu, covariance = estimate_window(history)
            if strategy == "single_period":
                target = solve_target_weights(mu, covariance, held, 1, gamma, 0.0)
            elif strategy == "minimum_variance":
                target = solve_target_weights(
                    np.zeros(n_assets), covariance, held, 1, gamma, 0.0
                )
            elif strategy == "multi_period":
                target = solve_target_weights(mu, covariance, held, horizon, gamma, penalty)
            else:
                target = predict_weights(trained_policy, history, held)

        trades = target - held
        turnover = float(np.abs(trades).sum())
        cost = cost_rate * turnover
        gross = float(target @ realized)
        if 1.0 + gross <= 0.0:
            raise ValueError("portfolio value hit zero; returns are implausible for this test")

        records.append(
            {
                "date": returns.index[index],
                "gross_return": gross,
                "net_return": gross - cost,
                "turnover": turnover,
                "cost": cost,
                "zero_trade_share": float(np.mean(np.abs(trades) <= 1e-6)),
            }
        )
        held = target * (1.0 + realized) / (1.0 + gross)

    return pd.DataFrame(records).set_index("date")


def summarize(result: pd.DataFrame, periods_per_year: int = PERIODS_PER_YEAR) -> pd.Series:
    """Net-of-cost performance metrics. Sharpe assumes a zero risk-free rate."""
    net = result["net_return"]
    equity = (1.0 + net).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    volatility = float(net.std(ddof=1) * np.sqrt(periods_per_year))
    drawdown = float((equity / equity.cummax() - 1.0).min())
    sharpe = (
        float(net.mean() / net.std(ddof=1) * np.sqrt(periods_per_year))
        if net.std(ddof=1) > 0.0
        else float("nan")
    )
    return pd.Series(
        {
            "periods": len(net),
            "total_net_return": total_return,
            "annualized_return": float(
                (1.0 + total_return) ** (periods_per_year / len(net)) - 1.0
            ),
            "annualized_volatility": volatility,
            "sharpe_net": sharpe,
            "max_drawdown": drawdown,
            "mean_turnover": float(result["turnover"].mean()),
            "total_cost": float(result["cost"].sum()),
            "zero_trade_fraction": float(result["zero_trade_share"].mean()),
        }
    )


def compare(returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Summary metrics for every strategy on the same windows."""
    return pd.DataFrame(
        {name: summarize(run_backtest(returns, name, **kwargs)) for name in STRATEGIES}
    ).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the portfolio walk-forward comparison")
    parser.add_argument("--csv", help="local returns or prices CSV")
    parser.add_argument("--kind", choices=("returns", "prices"), default="returns")
    parser.add_argument("--date-column", help="date column name; defaults to the first column")
    parser.add_argument("--periods", type=int, default=150)
    parser.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = parser.parse_args()

    returns = (
        load_returns_csv(args.csv, kind=args.kind, date_column=args.date_column)
        if args.csv
        else generate_synthetic_market_data()
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda value: f"{value:.4f}")
    print(f"cost {args.cost_bps:.0f} bps, penalty = charged cost = {lambda_from_cost(args.cost_bps):.5f}\n")
    print(compare(returns, periods=args.periods, cost_bps=args.cost_bps).T)
