# Multi-Period Mean-Variance Optimization

**Working title:** Multi-Period Asset Allocation via ADMM Proximal Splitting: Trading Sparsity
with L1 Turnover Penalties and Ledoit-Wolf Shrinkage

Reinforcement learning is named in the repository title but is not implemented yet. The current
contents are the deterministic optimizer described below.

## The problem

Single-period Markowitz optimization ignores trading friction. Run daily, it rebalances every
asset every day in response to noise, and the transaction costs eat the return. Adding an L1
turnover penalty `lambda * ||x_t - x_{t-1}||_1` over a multi-period horizon addresses this, but
the absolute value is non-differentiable and couples consecutive time steps, so a standard
quadratic solver either fails or never produces exact zero-trade days.

## Approach

### Stage 1: parameter estimation

- Expected returns: exponentially weighted moving average, to weight recent observations.
- Covariance: Ledoit-Wolf shrinkage toward a structured target, which keeps the matrix symmetric
  positive definite so the solver's matrix inversion stays well defined.

### Stage 2: ADMM

- Split the trade vector into an auxiliary variable `u_t = x_t - x_{t-1}`.
- Handle the L1 term with its proximal operator, soft-thresholding, which sets small trades to
  exactly zero rather than merely small.
- Project weights onto the simplex each iteration to keep the portfolio long-only and fully
  invested.

## Current baseline

`multi_period_admm.py` holds the implementation:

1. Deterministic correlated synthetic returns for smoke testing.
2. EWMA expected returns.
3. Positive-definite covariance matrices via Ledoit-Wolf shrinkage.
4. Long-only, fully invested multi-period ADMM optimization.
5. L1 soft-thresholding on trades to encourage sparsity.

Run it directly for a smoke run:

```text
python multi_period_admm.py
```

`LAG_driven_portfolio_optimization_baseline.ipynb` is a thin driver over the same module. It
checks convergence, simplex feasibility, and diversification, then reports allocation and turnover.

`LAG_driven_portfolio_oprimization.ipynb` is the original exploratory draft, including an LSTM
parameter engine. It is preserved for research history and is not maintained.

## Tests

Tests use the standard library, so no extra dependency is needed:

```text
python -m unittest -v
```

18 tests cover simplex projection, the soft-thresholding operator, estimator shapes and
positive-definiteness, ADMM convergence, the trade/weight consistency identity, penalty
monotonicity, and input validation.

## Setup

Use Python 3.11 or a compatible environment:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Add `pip install jupyter` to run the notebook.

## Verified baseline run

Executed on Python 3.11.9 with numpy 1.24.3, pandas 2.0.3, scikit-learn 1.3.0:

```text
periods_optimized          50
assets                     20
iterations                153
final_primal_residual      9.9e-05
final_dual_residual        3.9e-05
zero_trade_fraction        0.890
final_active_positions     20
final_effective_assets     14.70
```

Weights satisfy the long-only simplex constraint to machine precision (maximum sum error
2.2e-16), and every covariance matrix is positive definite (smallest eigenvalue 2.07e-05).

Risk aversion has to be calibrated to the data scale. Daily expected returns are order 1e-3 while
portfolio variance is order 1e-6, so a small `GAMMA` makes the linear return term dominate and the
optimizer collapses to a single asset. At `GAMMA = 5.0` the solution held one asset and needed
5,683 iterations; at `GAMMA = 1000.0` it holds 20 assets and converges in 153. The notebook
asserts both convergence and diversification, so a regression fails instead of printing a passing
summary.

Two caveats on these numbers. `GAMMA = 1000.0` was chosen because it diversifies on this fixture,
not derived from a volatility target. `LAMBDA = 0.01` is not tied to real trading costs, so
`zero_trade_fraction` describes the penalty setting rather than a cost saving.

## Research sequence

Next is a documented market-data adapter and a walk-forward backtest against equal-weight,
buy-and-hold, and single-period Markowitz benchmarks, measured net of transaction costs. The LSTM
parameter engine should return only after that, compared against Ledoit-Wolf on both out-of-sample
likelihood and portfolio metrics. Reinforcement learning comes last, as a comparison model.
