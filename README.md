# Multi-Period Portfolio Optimization

This repository develops a multi-period mean-variance portfolio optimizer with transaction-cost-aware trading. The current canonical baseline is deterministic and is intended to be correct and reproducible before deep learning or reinforcement learning is added.

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

`LAG_driven_portfolio_oprimization.ipynb` is the original exploratory draft. It is preserved for
research history and is not maintained.

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
pip install jupyter
```

Open the canonical notebook with Jupyter and run its cells from top to bottom. The reference ZIP archives in the project directory are local skill material and are intentionally excluded from Git.

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

Weights satisfy the long-only simplex constraint to machine precision (maximum sum error 2.2e-16), and every covariance matrix is positive definite (smallest eigenvalue 2.07e-05).

Risk aversion is calibrated to the data scale. Daily expected returns are order 1e-3 while portfolio variance is order 1e-6, so a small `GAMMA` makes the linear return term dominate and the optimizer collapses to a single asset. At `GAMMA = 5.0` the solution held one asset and needed 5,683 iterations; at `GAMMA = 1000.0` it holds 20 assets and converges in 153. The notebook asserts both convergence and diversification so a regression fails loudly instead of printing a passing summary.

## Research sequence

The next milestone is a local, documented market-data adapter and a walk-forward backtest against equal-weight and single-period benchmarks. The original LSTM parameter engine should be reintroduced only after the deterministic baseline has passing numerical checks and meaningful out-of-sample metrics. Reinforcement learning is a later comparison model, not part of the current baseline.
