"""Tests for the small parameter sensitivity helper."""

import unittest

import numpy as np

from multi_period_admm import generate_synthetic_market_data
from parameter_study import sensitivity_table


class TestSensitivityTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = generate_synthetic_market_data(n_assets=4, n_days=90)

    def test_varies_one_backtest_argument(self):
        table = sensitivity_table(
            self.returns,
            "multi_period",
            "rebalance_every",
            (1, 2),
            lookback=20,
            periods=8,
            horizon=2,
        )
        self.assertEqual(list(table.index.get_level_values("value")), [1, 2])
        self.assertTrue(np.isfinite(table["total_net_return"].to_numpy()).all())

    def test_rejects_unknown_parameter_and_empty_values(self):
        with self.assertRaises(ValueError):
            sensitivity_table(self.returns, "multi_period", "gamma", (1.0,))
        with self.assertRaises(ValueError):
            sensitivity_table(self.returns, "multi_period", "lookback", ())


if __name__ == "__main__":
    unittest.main()
