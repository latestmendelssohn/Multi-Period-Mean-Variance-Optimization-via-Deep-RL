"""Small policy-gradient portfolio benchmark.

The agent is intentionally simple: a linear policy maps recent return statistics and current
weights to softmax portfolio weights. During training, Gaussian logit noise provides exploration
and the reward includes linear turnover cost. The backtest trains it once before the test window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LOOKBACK = 20
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EXPLORATION = 0.10
DEFAULT_SEED = 0


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def _validate_weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_assets,) or not np.isclose(weights.sum(), 1.0) or np.any(weights < 0.0):
        raise ValueError("initial_weights must be a long-only simplex vector")
    return weights


def _state(history: pd.DataFrame, current_weights: np.ndarray) -> np.ndarray:
    values = history.to_numpy(dtype=float)
    mean = values.mean(axis=0) * 100.0
    volatility = values.std(axis=0) * 100.0
    return np.concatenate((mean, volatility, current_weights))


def train_policy(
    returns: pd.DataFrame,
    initial_weights: np.ndarray,
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    exploration: float = DEFAULT_EXPLORATION,
    cost_bps: float = 10.0,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Train a linear REINFORCE policy on returns available before the test window."""
    if returns.ndim != 2 or returns.shape[1] == 0:
        raise ValueError("returns must be a non-empty two-dimensional frame")
    if lookback < 2 or lookback >= len(returns):
        raise ValueError("lookback must be at least 2 and smaller than the number of rows")
    if epochs < 1 or learning_rate <= 0.0 or exploration <= 0.0 or cost_bps < 0.0:
        raise ValueError("epochs must be positive; learning rate, exploration, and cost must be non-negative")

    n_assets = returns.shape[1]
    held = _validate_weights(initial_weights, n_assets).copy()
    state_size = 3 * n_assets
    policy = np.zeros((n_assets, state_size))
    rng = np.random.default_rng(seed)
    cost_rate = cost_bps / 10_000.0

    for _ in range(epochs):
        held = _validate_weights(initial_weights, n_assets).copy()
        baseline = 0.0
        for index in range(lookback, len(returns)):
            realized = returns.iloc[index].to_numpy(dtype=float)
            state = _state(returns.iloc[index - lookback : index], held)
            noise = rng.normal(0.0, exploration, size=n_assets)
            action = _softmax(policy @ state + noise)
            turnover = float(np.abs(action - held).sum())
            gross = float(action @ realized)
            reward = gross - cost_rate * turnover

            advantage = reward - baseline
            baseline = 0.95 * baseline + 0.05 * reward
            score = noise / (exploration**2)
            policy += learning_rate * 100.0 * advantage * np.outer(score, state)
            policy = np.clip(policy, -10.0, 10.0)

            if 1.0 + gross <= 0.0:
                raise ValueError("portfolio value hit zero; returns are implausible for this test")
            held = action * (1.0 + realized) / (1.0 + gross)

    return policy


def predict_weights(
    policy: np.ndarray, history: pd.DataFrame, current_weights: np.ndarray
) -> np.ndarray:
    """Return a deterministic simplex action from a trained policy."""
    n_assets = history.shape[1]
    current_weights = _validate_weights(current_weights, n_assets)
    policy = np.asarray(policy, dtype=float)
    if policy.shape != (n_assets, 3 * n_assets):
        raise ValueError("policy has an incompatible shape")
    return _softmax(policy @ _state(history, current_weights))
