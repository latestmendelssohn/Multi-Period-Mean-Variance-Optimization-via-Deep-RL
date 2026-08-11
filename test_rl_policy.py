"""Tests for the small NumPy policy-gradient portfolio strategy."""

import unittest

import numpy as np

from backtest import run_backtest
from multi_period_admm import generate_synthetic_market_data
from rl_policy import predict_weights, train_policy


class TestRlPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = generate_synthetic_market_data(n_assets=6, n_days=140)
        cls.initial = np.full(6, 1.0 / 6.0)

    def test_training_is_deterministic(self):
        history = self.returns.iloc[:60]
        first = train_policy(history, self.initial, lookback=20, epochs=3, seed=7)
        second = train_policy(history, self.initial, lookback=20, epochs=3, seed=7)
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_prediction_is_a_simplex_action(self):
        history = self.returns.iloc[:60]
        policy = train_policy(history, self.initial, lookback=20, epochs=3, seed=7)
        action = predict_weights(policy, history.iloc[-20:], self.initial)
        self.assertEqual(action.shape, self.initial.shape)
        self.assertAlmostEqual(action.sum(), 1.0, places=12)
        self.assertTrue(np.all(action >= 0.0))

    def test_backtest_runs_the_rl_strategy(self):
        result = run_backtest(self.returns, "rl_policy", lookback=40, periods=12)
        self.assertEqual(len(result), 12)
        self.assertTrue(np.isfinite(result["net_return"].to_numpy()).all())
        self.assertTrue((result["turnover"] >= 0.0).all())

    def test_future_returns_do_not_change_past_rl_decisions(self):
        options = {"lookback": 40, "periods": 12}
        baseline = run_backtest(self.returns, "rl_policy", **options)
        tampered_returns = self.returns.copy()
        tampered_returns.iloc[-3:] = 0.5
        tampered = run_backtest(tampered_returns, "rl_policy", **options)
        np.testing.assert_allclose(
            baseline["turnover"].to_numpy()[:-3],
            tampered["turnover"].to_numpy()[:-3],
            atol=1e-12,
        )

    def test_rejects_insufficient_training_history(self):
        with self.assertRaises(ValueError):
            train_policy(self.returns.iloc[:2], self.initial, lookback=2)


if __name__ == "__main__":
    unittest.main()
