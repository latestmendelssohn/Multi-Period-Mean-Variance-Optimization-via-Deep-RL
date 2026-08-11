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

62 tests. The solver side covers simplex projection, the soft-thresholding operator, estimator
shapes and positive-definiteness, ADMM convergence, the trade/weight consistency identity, penalty
monotonicity, gamma calibration, and input validation. The backtest side covers cost accounting,
turnover, weight drift, the metrics, the cost-derived penalty, and a guard that fails if a decision
ever depends on data from after the moment it was made. The adapter side covers price and return
files plus every validation failure.

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

## Walk-forward backtest

`backtest.py` compares four strategies on the same windows, net of a linear transaction cost:

- `buy_and_hold` — buy equal weight once, never trade again.
- `equal_weight` — rebalance to equal weight every period.
- `single_period` — Markowitz re-solved every period with no turnover penalty.
- `multi_period` — the ADMM solver over a 5-period horizon, executing only the first step.

Every strategy estimates parameters from returns strictly before the period it trades into, and
all start from the same untraded equal-weight position. Parameters are held constant across the
horizon because future estimates do not exist at decision time. The L1 penalty inside the objective
defaults to the cost actually charged, so the optimizer is not avoiding a cost it never pays.

```text
python backtest.py
```

### Measured results

150 periods of the synthetic fixture, 10 bps per unit turnover, penalty equal to that cost:

```text
                       buy_and_hold  equal_weight  single_period  multi_period
total_net_return            -0.0350       -0.0320        -0.0550       -0.0420
annualized_return           -0.0581       -0.0531        -0.0906       -0.0696
annualized_volatility        0.0426        0.0423         0.0259        0.0257
sharpe_net                  -1.3840       -1.2680        -3.6483       -2.7908
max_drawdown                -0.0458       -0.0440        -0.0572       -0.0450
mean_turnover                0.0000        0.0086         0.1360        0.0738
total_cost                   0.0000        0.0013         0.0204        0.0111
zero_trade_fraction          1.0000        0.0080         0.0777        0.3707
```

Sweeping the cost, with the penalty tracking it:

```text
cost     penalty   single_period          multi_period
                   return   turnover      return   turnover   zero trades
0 bps    0.0000   -0.0355     0.1360     -0.0355     0.1359        0.0780
10 bps   0.0010   -0.0550     0.1360     -0.0420     0.0738        0.3707
50 bps   0.0050   -0.1291     0.1360     -0.0392     0.0210        0.7613
```

Three things follow from this table. At zero cost the penalty vanishes and the multi-period solver
reproduces single-period Markowitz to four decimal places, which is the reduction the model
predicts. As the cost rises the solver trades less on its own, from 0.1359 turnover down to 0.0210,
without any parameter being retuned. And single-period Markowitz gives up 0.0936 of return between
0 and 50 bps while the penalized solver gives up 0.0037.

An earlier version of this file reported `zero_trade_fraction` of 0.8863 and a net return of
-0.0272 at 10 bps. Those came from `LAMBDA = 0.01`, a penalty appropriate to a 100 bps cost while
only 10 bps was charged, so the optimizer was avoiding a cost it never paid. The numbers above use
a penalty equal to the charged cost and are lower as a result.

Three caveats. Every strategy loses money on this fixture, because the synthetic generator has
zero-mean factor loadings and no risk premium, so these are relative comparisons and not evidence
of profitability. The Sharpe ratios should not be ranked while returns are negative, since lower
volatility pushes a negative Sharpe further down. And synthetic returns cannot support any claim
about real markets.

### Risk aversion from a volatility target

`calibrate_gamma` bisects for the risk aversion whose single-period solution hits a target
annualized volatility, so the parameter follows from a stated risk appetite instead of being chosen
by hand. Calibrating to a 4 percent target on this fixture returns `gamma = 284.7`, and both
optimizers do worse with it than with the hand-picked `GAMMA = 1000.0`:

```text
                       single_period  multi_period
total_net_return             -0.0942       -0.0644
annualized_volatility         0.0366        0.0358
mean_turnover                 0.3095        0.1659
```

That is the honest comparison. The hand-picked value was chosen after seeing this fixture, and it
flatters the results; the calibrated value is reproducible on data the parameter was not fitted to.

## Real data

`data.py` loads returns from a local CSV, either as period returns or as price levels:

```python
from data import load_returns_csv
from backtest import compare

returns = load_returns_csv("prices.csv", kind="prices")
print(compare(returns, periods=150))
```

It validates numeric columns, missing values, duplicate and unsorted dates, and non-positive
prices, so a malformed file fails at load instead of producing plausible-looking results. Fetching
data over the network would add a dependency and is left out deliberately.

## Research sequence

Next is a documented market-data adapter, so the backtest runs on real returns instead of a
synthetic fixture, and a cost model expressed in basis points rather than the current abstract
`LAMBDA`. `GAMMA` should follow from a volatility target instead of being chosen for this fixture.
The LSTM parameter engine comes after that, compared against Ledoit-Wolf on both out-of-sample
likelihood and net portfolio metrics. Reinforcement learning comes last, as a comparison model.
