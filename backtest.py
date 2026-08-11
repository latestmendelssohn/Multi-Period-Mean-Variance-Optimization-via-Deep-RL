"""Walk-forward backtest for the multi-period optimizer and its benchmarks.

Every strategy sees only returns strictly before the period it trades into, holds a long-only
fully invested portfolio, and pays a linear cost on traded notional. All strategies start from
an equal-weight position, which is not charged, so the comparison is not distorted by an
arbitrary initial trade.

Run `python backtest.py` for the comparison table on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multi_period_admm import (
    GAMMA,
    LAMBDA,
    LOOKBACK,
    admm_multi_period_optimizer,
    estimate_window,
    generate_synthetic_market_data,
)

COST_BPS = 10.0
HORIZON = 5
PERIODS_PER_YEAR = 252
STRATEGIES = ("buy_and_hold", "equal_weight", "single_period", "multi_period")


def _solve_target(
    mu: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    horizon: int,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    """Target weights for the next period.

    Parameters are held constant across the horizon because future estimates are not
    available at decision time. Only the first period of the solved path is executed.
    """
    mu_sequence = np.tile(mu, (horizon, 1))
    sigma_sequence = np.tile(covariance, (horizon, 1, 1))
    weights, _, _ = admm_multi_period_optimizer(
        mu_sequence,
        sigma_sequence,
        current_weights,
        gamma=gamma,
        lambda_=lambda_,
        max_iter=2000,
        tolerance=1e-6,
    )
    return weights[0]


def run_backtest(
    returns: pd.DataFrame,
    strategy: str,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    gamma: float = GAMMA,
    lambda_: float = LAMBDA,
    cost_bps: float = COST_BPS,
    periods: int | None = None,
) -> pd.DataFrame:
    """Run one strategy and return per-period gross return, net return, turnover, and cost."""
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

    cost_rate = cost_bps / 10_000.0
    held = np.full(n_assets, 1.0 / n_assets)
    records = []

    for index in range(first, len(returns)):
        history = returns.iloc[index - lookback : index]
        realized = returns.iloc[index].to_numpy()

        if strategy == "buy_and_hold":
            target = held
        elif strategy == "equal_weight":
            target = np.full(n_assets, 1.0 / n_assets)
        else:
            mu, covariance = estimate_window(history)
            if strategy == "single_period":
                target = _solve_target(mu, covariance, held, 1, gamma, 0.0)
            else:
                target = _solve_target(mu, covariance, held, horizon, gamma, lambda_)

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
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda value: f"{value:.4f}")
    table = compare(generate_synthetic_market_data(), periods=150)
    print(f"cost assumption: {COST_BPS:.0f} bps per unit turnover\n")
    print(table.T)
