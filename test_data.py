"""Tests for the CSV return adapter. Run: python -m unittest -v"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from data import load_returns_csv, prices_to_returns


class CsvTestCase(unittest.TestCase):
    def write_csv(self, text: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name


class TestLoadReturns(CsvTestCase):
    def test_reads_returns(self):
        path = self.write_csv("date,A,B\n2024-01-02,0.01,-0.02\n2024-01-03,0.00,0.03\n")
        returns = load_returns_csv(path)
        self.assertEqual(returns.shape, (2, 2))
        self.assertEqual(list(returns.columns), ["A", "B"])
        self.assertEqual(returns.index.name, "date")
        self.assertIsInstance(returns.index, pd.DatetimeIndex)
        np.testing.assert_allclose(returns.iloc[0].to_numpy(), [0.01, -0.02])

    def test_converts_prices(self):
        path = self.write_csv("date,A\n2024-01-02,100\n2024-01-03,110\n2024-01-04,99\n")
        returns = load_returns_csv(path, kind="prices")
        self.assertEqual(len(returns), 2)
        np.testing.assert_allclose(returns["A"].to_numpy(), [0.10, -0.10], atol=1e-12)

    def test_named_date_column(self):
        path = self.write_csv("A,when\n0.01,2024-01-02\n0.02,2024-01-03\n")
        returns = load_returns_csv(path, date_column="when")
        self.assertEqual(list(returns.columns), ["A"])
        self.assertEqual(len(returns), 2)

    def test_output_feeds_the_estimator(self):
        rows = "\n".join(
            f"2024-01-{day:02d},{0.001 * day:.4f},{-0.001 * day:.4f}" for day in range(1, 21)
        )
        path = self.write_csv(f"date,A,B\n{rows}\n")
        returns = load_returns_csv(path)
        from multi_period_admm import estimate_window

        mu, covariance = estimate_window(returns)
        self.assertEqual(mu.shape, (2,))
        self.assertEqual(covariance.shape, (2, 2))


class TestValidation(CsvTestCase):
    def test_rejects_unknown_kind(self):
        path = self.write_csv("date,A\n2024-01-02,0.01\n2024-01-03,0.02\n")
        with self.assertRaises(ValueError):
            load_returns_csv(path, kind="levels")

    def test_rejects_single_column(self):
        path = self.write_csv("date\n2024-01-02\n")
        with self.assertRaises(ValueError):
            load_returns_csv(path)

    def test_rejects_missing_date_column(self):
        path = self.write_csv("date,A\n2024-01-02,0.01\n2024-01-03,0.02\n")
        with self.assertRaises(ValueError):
            load_returns_csv(path, date_column="timestamp")

    def test_rejects_non_numeric_values(self):
        path = self.write_csv("date,A\n2024-01-02,oops\n2024-01-03,0.02\n")
        with self.assertRaises(ValueError):
            load_returns_csv(path)

    def test_rejects_missing_values(self):
        path = self.write_csv("date,A,B\n2024-01-02,0.01,\n2024-01-03,0.02,0.01\n")
        with self.assertRaises(ValueError):
            load_returns_csv(path)

    def test_rejects_duplicate_dates(self):
        path = self.write_csv("date,A\n2024-01-02,0.01\n2024-01-02,0.02\n")
        with self.assertRaises(ValueError):
            load_returns_csv(path)

    def test_rejects_unsorted_dates(self):
        path = self.write_csv("date,A\n2024-01-03,0.01\n2024-01-02,0.02\n")
        with self.assertRaises(ValueError):
            load_returns_csv(path)

    def test_rejects_too_few_periods(self):
        path = self.write_csv("date,A\n2024-01-02,0.01\n")
        with self.assertRaises(ValueError):
            load_returns_csv(path)

    def test_rejects_non_positive_prices(self):
        path = self.write_csv("date,A\n2024-01-02,100\n2024-01-03,0\n2024-01-04,50\n")
        with self.assertRaises(ValueError):
            load_returns_csv(path, kind="prices")


class TestPricesToReturns(unittest.TestCase):
    def test_known_values(self):
        prices = pd.DataFrame({"A": [100.0, 105.0, 94.5]})
        np.testing.assert_allclose(
            prices_to_returns(prices)["A"].to_numpy(), [0.05, -0.10], atol=1e-12
        )

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            prices_to_returns(pd.DataFrame({"A": [100.0, -1.0]}))


if __name__ == "__main__":
    unittest.main()
