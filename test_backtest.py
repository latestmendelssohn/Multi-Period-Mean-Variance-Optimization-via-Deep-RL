"""Tests for the walk-forward backtest. Run: python -m unittest -v"""

import unittest

import numpy as np
import pandas as pd

from backtest import COST_BPS, compare, run_backtest, summarize
from multi_period_admm import generate_synthetic_market_data, lambda_from_cost

RETURNS = generate_synthetic_market_data(n_assets=6, n_days=140)
LOOKBACK = 40
PERIODS = 12


def _run(strategy, **kwargs):
    options = {"lookback": LOOKBACK, "periods": PERIODS, "horizon": 3}
    options.update(kwargs)
    return run_backtest(RETURNS, strategy, **options)


class TestNoLookAhead(unittest.TestCase):
    """The decisive property: a decision may not depend on data after it was made."""

    def test_future_returns_cannot_change_past_decisions(self):
        for strategy in ("equal_weight", "single_period", "multi_period"):
            with self.subTest(strategy=strategy):
                baseline = _run(strategy)
                tampered_returns = RETURNS.copy()
                # Replace the final three periods with extreme values.
                tampered_returns.iloc[-3:] = 0.5
                tampered = run_backtest(
                    tampered_returns, strategy, lookback=LOOKBACK, periods=PERIODS, horizon=3
                )
                # Every period before the tampered tail must be identical.
                np.testing.assert_allclose(
                    baseline["turnover"].to_numpy()[:-3],
                    tampered["turnover"].to_numpy()[:-3],
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    baseline["gross_return"].to_numpy()[:-4],
                    tampered["gross_return"].to_numpy()[:-4],
                    atol=1e-12,
                )

    def test_truncating_history_does_not_change_earlier_periods(self):
        full = _run("multi_period", periods=PERIODS)
        shorter = run_backtest(
            RETURNS.iloc[:-2], "multi_period", lookback=LOOKBACK, periods=PERIODS - 2, horizon=3
        )
        overlap = min(len(full) - 2, len(shorter))
        np.testing.assert_allclose(
            full["net_return"].to_numpy()[:overlap],
            shorter["net_return"].to_numpy()[:overlap],
            atol=1e-12,
        )


class TestCostsAndTurnover(unittest.TestCase):
    def test_buy_and_hold_never_trades(self):
        result = _run("buy_and_hold")
        np.testing.assert_allclose(result["turnover"].to_numpy(), 0.0, atol=1e-12)
        self.assertEqual(result["cost"].sum(), 0.0)
        np.testing.assert_allclose(
            result["net_return"].to_numpy(), result["gross_return"].to_numpy(), atol=1e-12
        )

    def test_net_equals_gross_minus_cost(self):
        result = _run("equal_weight")
        np.testing.assert_allclose(
            result["net_return"].to_numpy(),
            (result["gross_return"] - result["cost"]).to_numpy(),
            atol=1e-15,
        )

    def test_cost_matches_turnover_times_rate(self):
        result = _run("equal_weight")
        np.testing.assert_allclose(
            result["cost"].to_numpy(),
            result["turnover"].to_numpy() * COST_BPS / 10_000.0,
            atol=1e-15,
        )

    def test_higher_cost_never_improves_net_return(self):
        cheap = summarize(_run("equal_weight", cost_bps=1.0))["total_net_return"]
        expensive = summarize(_run("equal_weight", cost_bps=100.0))["total_net_return"]
        self.assertGreater(cheap, expensive)

    def test_zero_cost_leaves_gross_unchanged(self):
        result = _run("equal_weight", cost_bps=0.0)
        np.testing.assert_allclose(
            result["net_return"].to_numpy(), result["gross_return"].to_numpy(), atol=1e-15
        )

    def test_turnover_penalty_reduces_trading(self):
        low = summarize(_run("multi_period", lambda_=0.0))["mean_turnover"]
        high = summarize(_run("multi_period", lambda_=0.05))["mean_turnover"]
        self.assertLessEqual(high, low)


class TestPortfolioMechanics(unittest.TestCase):
    def test_equal_weight_targets_uniform_weights(self):
        # The first period already starts at equal weight, so it trades nothing. After that,
        # turnover only corrects the previous period's drift and stays small.
        result = _run("equal_weight")
        turnover = result["turnover"].to_numpy()
        self.assertAlmostEqual(turnover[0], 0.0, places=12)
        self.assertTrue((turnover[1:] > 0.0).all())
        self.assertLess(turnover.max(), 0.5)

    def test_all_periods_are_covered(self):
        result = _run("equal_weight")
        self.assertEqual(len(result), PERIODS)
        self.assertTrue(result.index.equals(RETURNS.index[-PERIODS:]))

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            _run("nonexistent_strategy")
        with self.assertRaises(ValueError):
            _run("equal_weight", lookback=1)
        with self.assertRaises(ValueError):
            _run("equal_weight", lookback=len(RETURNS))
        with self.assertRaises(ValueError):
            _run("equal_weight", cost_bps=-1.0)
        with self.assertRaises(ValueError):
            _run("equal_weight", periods=0)


class TestSummarize(unittest.TestCase):
    def test_constant_return_series(self):
        frame = pd.DataFrame(
            {
                "gross_return": [0.001] * 252,
                "net_return": [0.001] * 252,
                "turnover": [0.0] * 252,
                "cost": [0.0] * 252,
                "zero_trade_share": [1.0] * 252,
            }
        )
        metrics = summarize(frame)
        self.assertAlmostEqual(metrics["total_net_return"], 1.001**252 - 1, places=10)
        self.assertAlmostEqual(metrics["annualized_return"], 1.001**252 - 1, places=10)
        self.assertAlmostEqual(metrics["annualized_volatility"], 0.0, places=12)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.0, places=12)

    def test_max_drawdown_on_known_path(self):
        frame = pd.DataFrame(
            {
                "gross_return": [0.0, -0.5, 0.0],
                "net_return": [0.0, -0.5, 0.0],
                "turnover": [0.0] * 3,
                "cost": [0.0] * 3,
                "zero_trade_share": [1.0] * 3,
            }
        )
        self.assertAlmostEqual(summarize(frame)["max_drawdown"], -0.5, places=12)


class TestCompare(unittest.TestCase):
    def test_all_strategies_reported_on_same_windows(self):
        table = compare(RETURNS, lookback=LOOKBACK, periods=8, horizon=3)
        self.assertEqual(len(table), 5)
        self.assertTrue((table["periods"] == 8).all())
        self.assertTrue(np.isfinite(table["total_net_return"].to_numpy()).all())


class TestCostConsistentPenalty(unittest.TestCase):
    def test_default_penalty_equals_the_charged_cost(self):
        derived = _run("multi_period", cost_bps=25.0)
        explicit = _run("multi_period", cost_bps=25.0, lambda_=lambda_from_cost(25.0))
        np.testing.assert_allclose(
            derived["turnover"].to_numpy(), explicit["turnover"].to_numpy(), atol=1e-12
        )

    def test_explicit_penalty_overrides_the_default(self):
        derived = _run("multi_period", cost_bps=10.0)
        overridden = _run("multi_period", cost_bps=10.0, lambda_=0.05)
        self.assertLessEqual(
            overridden["turnover"].mean(), derived["turnover"].mean()
        )

    def test_rejects_negative_penalty(self):
        with self.assertRaises(ValueError):
            _run("multi_period", lambda_=-0.01)


if __name__ == "__main__":
    unittest.main()
