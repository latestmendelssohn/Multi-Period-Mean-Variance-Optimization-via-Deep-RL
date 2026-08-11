"""Tests for the multi-period ADMM optimizer. Run: python -m unittest -v"""

import unittest

import numpy as np

from multi_period_admm import (
    admm_multi_period_optimizer,
    annualized_volatility,
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
        # Needs a generous iteration budget: forcing a near-corner solution makes the dual
        # residual fall slowly.
        mu, sigma, x0 = _tiny_problem()
        weights, _, _ = admm_multi_period_optimizer(
            mu, sigma, x0, gamma=1e-3, lambda_=0.0, max_iter=100000, tolerance=1e-8
        )
        effective_assets = 1.0 / np.sum(weights[-1] ** 2)
        self.assertLess(effective_assets, 2.0)


def _reference_single_period(mu, covariance, gamma, iterations=50000):
    """Projected gradient ascent on the single-period objective.

    Deliberately a different algorithm from the one under test. Projection onto the simplex is
    exact, so this converges to the constrained optimum and can referee the ADMM solution.
    """
    n_assets = len(mu)
    weights = np.full(n_assets, 1.0 / n_assets)
    step = 1.0 / (gamma * np.linalg.eigvalsh(covariance).max() + 1e-9)
    for _ in range(iterations):
        weights = project_simplex(weights + step * (mu - gamma * covariance @ weights))
    return weights


class TestOptimality(unittest.TestCase):
    """Guards the defect where the solver projected the unconstrained optimum onto the simplex.

    That shortcut looked reasonable and stayed accurate at low risk aversion, but its error grew
    with gamma until the solution collapsed to equal weights instead of minimum variance.
    """

    @classmethod
    def setUpClass(cls):
        returns = generate_synthetic_market_data(n_assets=6, n_days=160)
        cls.mu, cls.covariance = estimate_window(returns.iloc[-60:])
        cls.equal = np.full(6, 1.0 / 6.0)

    def _objective(self, weights, gamma):
        return float(self.mu @ weights - 0.5 * gamma * (weights @ self.covariance @ weights))

    def _solve(self, gamma):
        return solve_target_weights(
            self.mu, self.covariance, self.equal, gamma=gamma, max_iter=8000, tolerance=1e-9
        )

    def _relative_gap(self, gamma):
        reference = self._objective(_reference_single_period(self.mu, self.covariance, gamma), gamma)
        return (reference - self._objective(self._solve(gamma), gamma)) / max(abs(reference), 1e-12)

    def test_matches_reference_where_risk_matters(self):
        for gamma in (100.0, 1_000.0, 10_000.0, 1_000_000.0):
            with self.subTest(gamma=gamma):
                self.assertLess(self._relative_gap(gamma), 1e-5)

    def test_near_linear_objective_is_close_but_slower(self):
        # With gamma this low the risk term barely matters and the optimum sits on a vertex of
        # the simplex, which ADMM approaches slowly. Still close, but not to solver precision.
        self.assertLess(self._relative_gap(1.0), 1e-3)

    def test_high_risk_aversion_approaches_minimum_variance(self):
        # The failure mode being guarded against returned equal weights here, whose variance is
        # far above the minimum.
        solved = self._solve(1_000_000.0)
        reference = _reference_single_period(self.mu, self.covariance, 1_000_000.0)
        solved_variance = float(solved @ self.covariance @ solved)
        equal_variance = float(self.equal @ self.covariance @ self.equal)
        self.assertLess(solved_variance, equal_variance)
        self.assertAlmostEqual(
            solved_variance / float(reference @ self.covariance @ reference), 1.0, delta=0.01
        )

    def test_volatility_falls_as_risk_aversion_rises(self):
        volatilities = [
            annualized_volatility(self._solve(gamma), self.covariance)
            for gamma in (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
        ]
        for lower, higher in zip(volatilities[1:], volatilities[:-1]):
            self.assertLessEqual(lower, higher + 1e-6)

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


if __name__ == "__main__":
    unittest.main()
