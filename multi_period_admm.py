"""Multi-period mean-variance optimization with an L1 turnover penalty.

Estimators: EWMA expected returns, Ledoit-Wolf covariance.
Solver: ADMM with simplex projection for weights and soft-thresholding for trades.

Run `python multi_period_admm.py` for a deterministic smoke run on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

SEED = 42
N_ASSETS = 20
N_DAYS = 500
LOOKBACK = 60
HORIZON = 50
# Risk aversion is scaled to daily data: expected returns are order 1e-3 while portfolio
# variance is order 1e-6, so a small gamma degenerates to a single-asset corner solution.
GAMMA = 1000.0
# Penalty for the standalone smoke run only. 0.01 in return units is the cost of trading at
# 100 bps, so it is deliberately stricter than any cost charged here. The backtest derives its
# penalty from the cost it actually charges; see lambda_from_cost.
LAMBDA = 0.01
RHO = 1.0
TOLERANCE = 1e-4
RIDGE_LAGS = 5
RIDGE_ALPHA = 1.0


def generate_synthetic_market_data(
    n_assets: int = N_ASSETS, n_days: int = N_DAYS, seed: int = SEED
) -> pd.DataFrame:
    """Deterministic correlated returns from a linear factor model. Test fixture only."""
    rng = np.random.default_rng(seed)
    n_factors = min(3, n_assets)
    factor_returns = rng.normal(0.0002, 0.01, size=(n_days, n_factors))
    loadings = rng.normal(0.0, 0.5, size=(n_assets, n_factors))
    idiosyncratic = rng.normal(0.0, 0.006, size=(n_days, n_assets))
    returns = factor_returns @ loadings.T + idiosyncratic
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    columns = [f"Asset_{index}" for index in range(n_assets)]
    return pd.DataFrame(returns, index=dates, columns=columns)


def estimate_window(
    window: pd.DataFrame, forecast_method: str = "ewma"
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate expected returns and covariance for one lookback window."""
    if forecast_method == "ewma":
        span = max(2, len(window) // 2)
        mu = window.ewm(span=span, adjust=False).mean().iloc[-1].to_numpy()
    elif forecast_method == "lag1":
        values = window.to_numpy(dtype=float)
        design = np.column_stack((np.ones(len(values) - 1), values[:-1]))
        coefficients = np.linalg.lstsq(design, values[1:], rcond=None)[0]
        mu = coefficients[0] + values[-1] @ coefficients[1:]
    elif forecast_method == "ridge":
        # Per-asset Ridge on the trailing window using RIDGE_LAGS previous returns as features.
        # Regularization keeps the estimate stable when the lookback is short.
        values = window.to_numpy(dtype=float)
        n_rows = len(values)
        if n_rows <= RIDGE_LAGS + 1:
            raise ValueError("ridge forecast needs more rows than the number of lags")
        target = values[RIDGE_LAGS:]
        design = np.column_stack(
            [values[RIDGE_LAGS - 1 - lag : n_rows - 1 - lag] for lag in range(RIDGE_LAGS)]
        )
        recent = values[n_rows - RIDGE_LAGS : n_rows][::-1].reshape(1, -1)
        mu = np.empty(values.shape[1])
        for asset in range(values.shape[1]):
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(design, target[:, asset])
            mu[asset] = float(model.predict(recent)[0])
    else:
        raise ValueError("forecast_method must be 'ewma', 'lag1', or 'ridge'")

    covariance = LedoitWolf().fit(window.to_numpy()).covariance_
    covariance = (covariance + covariance.T) / 2.0
    return mu, covariance + 1e-10 * np.eye(covariance.shape[0])


def estimate_parameters(
    returns: pd.DataFrame, lookback: int = LOOKBACK, forecast_method: str = "ewma"
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling expected returns and Ledoit-Wolf covariances after each window."""
    if lookback < 2 or lookback >= len(returns):
        raise ValueError("lookback must be at least 2 and smaller than the number of rows")

    n_periods, n_assets = returns.shape
    mu_sequence = np.empty((n_periods - lookback, n_assets))
    sigma_sequence = np.empty((n_periods - lookback, n_assets, n_assets))

    for period in range(lookback, n_periods):
        mu, covariance = estimate_window(
            returns.iloc[period - lookback : period], forecast_method=forecast_method
        )
        mu_sequence[period - lookback] = mu
        sigma_sequence[period - lookback] = covariance

    return mu_sequence, sigma_sequence


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {x : sum(x) = 1, x >= 0}."""
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("vector must be a non-empty one-dimensional array")
    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values)
    active = np.nonzero(sorted_values * np.arange(1, vector.size + 1) > cumulative - 1.0)[0]
    if active.size == 0:
        raise ValueError("simplex projection failed to find an active set")
    threshold = (cumulative[active[-1]] - 1.0) / (active[-1] + 1.0)
    return np.maximum(vector - threshold, 0.0)


def soft_threshold(vector: np.ndarray, threshold: float) -> np.ndarray:
    """Proximal operator of the L1 norm."""
    if threshold < 0.0:
        raise ValueError("threshold cannot be negative")
    return np.sign(vector) * np.maximum(np.abs(vector) - threshold, 0.0)


def admm_multi_period_optimizer(
    mu_sequence: np.ndarray,
    sigma_sequence: np.ndarray,
    initial_weights: np.ndarray,
    gamma: float = GAMMA,
    lambda_: float = LAMBDA,
    rho: float = RHO,
    max_iter: int = 8000,
    tolerance: float = TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Solve the multi-period problem and return weights, trades, and the residual trace.

    Three splits. The weight block is an exact linear solve. Feasibility is enforced by a
    consensus copy whose update is the simplex projection, and turnover by an auxiliary trade
    variable whose update is soft-thresholding.

    Projecting the unconstrained minimizer onto the simplex directly would be cheaper but wrong:
    the projection of the unconstrained optimum is not the constrained optimum, and the error
    grows with `gamma` until the solution collapses to equal weights. The consensus copy is what
    makes high risk aversion converge to minimum variance instead.

    Convergence is not guaranteed within `max_iter`; check the last residual pair.
    """
    mu_sequence = np.asarray(mu_sequence, dtype=float)
    sigma_sequence = np.asarray(sigma_sequence, dtype=float)
    initial_weights = np.asarray(initial_weights, dtype=float)
    if mu_sequence.ndim != 2:
        raise ValueError("mu_sequence must have shape (periods, assets)")
    periods, n_assets = mu_sequence.shape
    if sigma_sequence.shape != (periods, n_assets, n_assets):
        raise ValueError("sigma_sequence must have shape (periods, assets, assets)")
    if initial_weights.shape != (n_assets,):
        raise ValueError("initial_weights must have shape (assets,)")
    if not np.isclose(initial_weights.sum(), 1.0) or np.any(initial_weights < 0.0):
        raise ValueError("initial_weights must be a long-only simplex vector")
    if gamma <= 0.0 or lambda_ < 0.0 or rho <= 0.0:
        raise ValueError("gamma and rho must be positive; lambda_ cannot be negative")

    identity = np.eye(n_assets)
    for covariance in sigma_sequence:
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("covariance matrices must be symmetric")
        if np.linalg.eigvalsh(covariance).min() <= 0.0:
            raise ValueError("covariance matrices must be positive definite")

    weights = np.tile(initial_weights, (periods, 1))
    feasible = weights.copy()
    trades = np.zeros((periods, n_assets))
    trade_dual = np.zeros((periods, n_assets))
    feasibility_dual = np.zeros((periods, n_assets))

    system_inverses = []
    for period, covariance in enumerate(sigma_sequence):
        coupling = 2.0 if period < periods - 1 else 1.0
        system_inverses.append(
            np.linalg.inv(gamma * covariance + (coupling + 1.0) * rho * identity)
        )

    residual_trace: list[tuple[float, float]] = []
    for _ in range(max_iter):
        old_trades = trades.copy()
        old_feasible = feasible.copy()

        for period in range(periods):
            previous = initial_weights if period == 0 else weights[period - 1]
            right_hand_side = (
                mu_sequence[period]
                + rho * (previous + trades[period] - trade_dual[period] / rho)
                + rho * (feasible[period] - feasibility_dual[period] / rho)
            )
            if period < periods - 1:
                right_hand_side += rho * (
                    weights[period + 1] - trades[period + 1] + trade_dual[period + 1] / rho
                )
            weights[period] = system_inverses[period] @ right_hand_side

        for period in range(periods):
            feasible[period] = project_simplex(
                weights[period] + feasibility_dual[period] / rho
            )

        for period in range(periods):
            previous = initial_weights if period == 0 else weights[period - 1]
            trades[period] = soft_threshold(
                weights[period] - previous + trade_dual[period] / rho, lambda_ / rho
            )

        for period in range(periods):
            previous = initial_weights if period == 0 else weights[period - 1]
            trade_dual[period] += rho * (weights[period] - previous - trades[period])
            feasibility_dual[period] += rho * (weights[period] - feasible[period])

        primal = np.sqrt(
            sum(
                np.linalg.norm(
                    weights[period]
                    - (initial_weights if period == 0 else weights[period - 1])
                    - trades[period]
                )
                ** 2
                for period in range(periods)
            )
            + np.linalg.norm(weights - feasible) ** 2
        )
        dual_residual = rho * np.sqrt(
            np.linalg.norm(trades - old_trades) ** 2
            + np.linalg.norm(feasible - old_feasible) ** 2
        )
        residual_trace.append((float(primal), float(dual_residual)))
        if primal < tolerance and dual_residual < tolerance:
            break

    return feasible, trades, residual_trace


def lambda_from_cost(cost_bps: float) -> float:
    """L1 penalty matching a linear cost of `cost_bps` per unit traded notional.

    The penalty and the charged cost are then the same number in return units, so the optimizer
    is penalized for the cost it actually pays rather than an unrelated constant.
    """
    if cost_bps < 0.0:
        raise ValueError("cost_bps cannot be negative")
    return cost_bps / 10_000.0


def solve_target_weights(
    mu: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    horizon: int = 1,
    gamma: float = GAMMA,
    lambda_: float = 0.0,
    max_iter: int = 2000,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Weights for the next period only.

    Parameters are held constant across the horizon, because future estimates do not exist at
    decision time. Only the first period of the solved path is returned.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    weights, _, _ = admm_multi_period_optimizer(
        np.tile(mu, (horizon, 1)),
        np.tile(covariance, (horizon, 1, 1)),
        current_weights,
        gamma=gamma,
        lambda_=lambda_,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    return weights[0]


def annualized_volatility(
    weights: np.ndarray, covariance: np.ndarray, periods_per_year: int = 252
) -> float:
    """Annualized volatility of a fixed weight vector under one covariance matrix."""
    variance = float(weights @ covariance @ weights)
    return float(np.sqrt(max(variance, 0.0) * periods_per_year))


def run_baseline() -> pd.Series:
    """Deterministic end-to-end run on synthetic data. Returns the summary metrics."""
    returns = generate_synthetic_market_data()
    mu_sequence, sigma_sequence = estimate_parameters(returns)
    mu_target = mu_sequence[-HORIZON:]
    sigma_target = sigma_sequence[-HORIZON:]
    initial_weights = np.full(mu_target.shape[1], 1.0 / mu_target.shape[1])

    weights, trades, residual_trace = admm_multi_period_optimizer(
        mu_target, sigma_target, initial_weights
    )
    final_primal, final_dual = residual_trace[-1]
    return pd.Series(
        {
            "periods_optimized": weights.shape[0],
            "assets": weights.shape[1],
            "iterations": len(residual_trace),
            "final_primal_residual": final_primal,
            "final_dual_residual": final_dual,
            "zero_trade_fraction": float(np.mean(np.abs(trades) <= 1e-6)),
            "final_active_positions": int(np.count_nonzero(weights[-1] > 1e-6)),
            "final_effective_assets": float(1.0 / np.sum(weights[-1] ** 2)),
        }
    )


if __name__ == "__main__":
    print(run_baseline())
