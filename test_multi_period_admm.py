"""Tests for the multi-period ADMM optimizer. Run: python -m unittest -v"""

import unittest

import numpy as np

from multi_period_admm import (
    admm_multi_period_optimizer,
    annualized_volatility,
    calibrate_gamma,
    estimate_parameters,
    estimate_window,
    generate_synthetic_market_data,
    lambda_from_cost,
    project_simplex,
    soft_threshold,
    solve_target_weights,
)


class TestProjectSimplex(unittest.TestCase):
    def test_valid_weights_are_unchanged(self):
        weights = np.array([0.5, 0.3, 0.2])
        np.testing.assert_allclose(project_simplex(weights), weights, atol=1e-12)

    def test_output_sums_to_one_and_is_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            projected = project_simplex(rng.normal(size=8) * 5.0)
            self.assertAlmostEqual(projected.sum(), 1.0, places=12)
            self.assertTrue(np.all(projected >= 0.0))

    def test_known_projection(self):
        # Shifting by a constant leaves the projection unchanged.
        base = np.array([0.6, 0.4, 0.0])
        np.testing.assert_allclose(project_simplex(base + 3.0), base, atol=1e-12)

    def test_dominant_component_takes_everything(self):
        projected = project_simplex(np.array([10.0, -5.0, -5.0]))
        np.testing.assert_allclose(projected, np.array([1.0, 0.0, 0.0]), atol=1e-12)

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            project_simplex(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            project_simplex(np.array([]))


class TestSoftThreshold(unittest.TestCase):
    def test_shrinks_towards_zero(self):
        result = soft_threshold(np.array([0.5, -0.5, 0.05, -0.05]), 0.1)
        np.testing.assert_allclose(result, np.array([0.4, -0.4, 0.0, 0.0]), atol=1e-12)

    def test_zero_threshold_is_identity(self):
        values = np.array([0.3, -1.2, 0.0])
        np.testing.assert_allclose(soft_threshold(values, 0.0), values, atol=1e-12)

    def test_rejects_negative_threshold(self):
        with self.assertRaises(ValueError):
            soft_threshold(np.array([1.0]), -0.1)


class TestEstimateParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = generate_synthetic_market_data(n_assets=5, n_days=120)
        cls.mu, cls.sigma = estimate_parameters(cls.returns, lookback=30)

    def test_shapes(self):
        self.assertEqual(self.mu.shape, (90, 5))
        self.assertEqual(self.sigma.shape, (90, 5, 5))

    def test_covariances_are_symmetric_positive_definite(self):
        for covariance in self.sigma:
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.0)

    def test_deterministic(self):
        mu_again, _ = estimate_parameters(
            generate_synthetic_market_data(n_assets=5, n_days=120), lookback=30
        )
        np.testing.assert_allclose(self.mu, mu_again)

    def test_rejects_bad_lookback(self):
        with self.assertRaises(ValueError):
            estimate_parameters(self.returns, lookback=1)
        with self.assertRaises(ValueError):
            estimate_parameters(self.returns, lookback=len(self.returns))


def _tiny_problem(periods=4, n_assets=3, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.001, 0.0005, size=(periods, n_assets))
    base = rng.normal(size=(n_assets, n_assets))
    covariance = base @ base.T / n_assets + np.eye(n_assets) * 0.01
    sigma = np.tile(covariance, (periods, 1, 1))
    return mu, sigma, np.full(n_assets, 1.0 / n_assets)


class TestAdmmOptimizer(unittest.TestCase):
    def test_converges_and_respects_constraints(self):
        mu, sigma, x0 = _tiny_problem()
        weights, trades, trace = admm_multi_period_optimizer(
            mu, sigma, x0, gamma=100.0, lambda_=0.001, tolerance=1e-6, max_iter=5000
        )
        primal, dual = trace[-1]
        self.assertLess(primal, 1e-6)
        self.assertLess(dual, 1e-6)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-8)
        self.assertTrue(np.all(weights >= -1e-10))
        self.assertTrue(np.isfinite(trades).all())

    def test_trades_match_weight_differences_at_convergence(self):
        mu, sigma, x0 = _tiny_problem()
        weights, trades, _ = admm_multi_period_optimizer(
            mu, sigma, x0, gamma=100.0, lambda_=0.0, tolerance=1e-8, max_iter=20000
        )
        previous = np.vstack([x0, weights[:-1]])
        np.testing.assert_allclose(trades, weights - previous, atol=1e-5)

    def test_higher_penalty_gives_at_least_as_many_zero_trades(self):
        mu, sigma, x0 = _tiny_problem()
        _, low_trades, _ = admm_multi_period_optimizer(
            mu, sigma, x0, gamma=100.0, lambda_=0.0, max_iter=2000
        )
        _, high_trades, _ = admm_multi_period_optimizer(
            mu, sigma, x0, gamma=100.0, lambda_=0.05, max_iter=2000
        )
        self.assertGreaterEqual(
            np.mean(np.abs(high_trades) <= 1e-6), np.mean(np.abs(low_trades) <= 1e-6)
        )

    def test_low_risk_aversion_concentrates_portfolio(self):
        mu, sigma, x0 = _tiny_problem()
        weights, _, _ = admm_multi_period_optimizer(
            mu, sigma, x0, gamma=1e-3, lambda_=0.0, max_iter=3000
        )
        effective_assets = 1.0 / np.sum(weights[-1] ** 2)
        self.assertLess(effective_assets, 2.0)

    def test_rejects_invalid_inputs(self):
        mu, sigma, x0 = _tiny_problem()
        with self.assertRaises(ValueError):
            admm_multi_period_optimizer(mu[0], sigma, x0)
        with self.assertRaises(ValueError):
            admm_multi_period_optimizer(mu, sigma[:, :, :1], x0)
        with self.assertRaises(ValueError):
            admm_multi_period_optimizer(mu, sigma, np.array([0.5, 0.5]))
        with self.assertRaises(ValueError):
            admm_multi_period_optimizer(mu, sigma, np.array([0.5, 0.6, 0.1]))
        with self.assertRaises(ValueError):
            admm_multi_period_optimizer(mu, sigma, x0, gamma=0.0)
        with self.assertRaises(ValueError):
            admm_multi_period_optimizer(mu, sigma, x0, lambda_=-1.0)

    def test_rejects_non_positive_definite_covariance(self):
        mu, sigma, x0 = _tiny_problem()
        broken = sigma.copy()
        broken[0] = np.zeros_like(broken[0])
        with self.assertRaises(ValueError):
            admm_multi_period_optimizer(mu, broken, x0)


class TestLambdaFromCost(unittest.TestCase):
    def test_converts_basis_points_to_return_units(self):
        self.assertAlmostEqual(lambda_from_cost(10.0), 0.001, places=12)
        self.assertAlmostEqual(lambda_from_cost(0.0), 0.0, places=12)

    def test_rejects_negative_cost(self):
        with self.assertRaises(ValueError):
            lambda_from_cost(-1.0)


class TestSolveTargetWeights(unittest.TestCase):
    def test_returns_one_valid_weight_vector(self):
        mu, sigma, x0 = _tiny_problem()
        weights = solve_target_weights(mu[0], sigma[0], x0, horizon=3, gamma=100.0)
        self.assertEqual(weights.shape, x0.shape)
        self.assertAlmostEqual(weights.sum(), 1.0, places=8)
        self.assertTrue(np.all(weights >= -1e-10))

    def test_rejects_bad_horizon(self):
        mu, sigma, x0 = _tiny_problem()
        with self.assertRaises(ValueError):
            solve_target_weights(mu[0], sigma[0], x0, horizon=0)


class TestCalibrateGamma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        returns = generate_synthetic_market_data(n_assets=6, n_days=160)
        cls.mu, cls.covariance = estimate_window(returns.iloc[-60:])
        cls.equal = np.full(6, 1.0 / 6.0)

    def _volatility_for(self, gamma):
        weights = solve_target_weights(self.mu, self.covariance, self.equal, gamma=gamma)
        return annualized_volatility(weights, self.covariance)

    def test_hits_a_reachable_target(self):
        reachable = (self._volatility_for(1e-2) + self._volatility_for(1e8)) / 2.0
        gamma = calibrate_gamma(self.mu, self.covariance, reachable, self.equal)
        self.assertAlmostEqual(self._volatility_for(gamma), reachable, delta=0.005)

    def test_higher_target_means_lower_risk_aversion(self):
        low_target = self._volatility_for(1e8) * 1.05
        high_target = self._volatility_for(1e-2) * 0.95
        self.assertLess(
            calibrate_gamma(self.mu, self.covariance, high_target, self.equal),
            calibrate_gamma(self.mu, self.covariance, low_target, self.equal),
        )

    def test_unreachable_targets_return_the_bounds(self):
        bounds = (1e-2, 1e8)
        self.assertEqual(
            calibrate_gamma(self.mu, self.covariance, 100.0, self.equal, bounds=bounds), 1e-2
        )
        self.assertEqual(
            calibrate_gamma(self.mu, self.covariance, 1e-9, self.equal, bounds=bounds), 1e8
        )

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            calibrate_gamma(self.mu, self.covariance, 0.0, self.equal)
        with self.assertRaises(ValueError):
            calibrate_gamma(self.mu, self.covariance, 0.05, self.equal, bounds=(10.0, 1.0))


class TestAnnualizedVolatility(unittest.TestCase):
    def test_known_value(self):
        covariance = np.diag([0.0001, 0.0001])
        weights = np.array([1.0, 0.0])
        self.assertAlmostEqual(
            annualized_volatility(weights, covariance, periods_per_year=252),
            np.sqrt(0.0001 * 252),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
