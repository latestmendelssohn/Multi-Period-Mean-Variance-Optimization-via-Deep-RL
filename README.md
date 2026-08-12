# Multi-period mean-variance optimization

A small portfolio-optimization project. It compares deterministic Markowitz and multi-period ADMM allocation with a lightweight policy-gradient reinforcement-learning benchmark.

## What is included

- EWMA expected returns, an optional lag-1 linear forecast, and Ledoit-Wolf covariance estimation.
- Long-only, fully invested portfolios: weights are non-negative and sum to one.
- Correct consensus ADMM updates for simplex feasibility and sparse trades.
- Linear transaction costs expressed in basis points, with cost-level comparisons.
- Configurable rebalancing frequency for the walk-forward backtest.
- A walk-forward backtest against buy-and-hold, equal weight, minimum variance, single-period Markowitz, multi-period ADMM, and `rl_policy`.
- A local CSV adapter for return or price files.
- A small NumPy-only REINFORCE-style policy. No PyTorch or deployment layer.

## Run

Create an environment and install the runtime dependencies:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the deterministic optimizer smoke test:

```text
python multi_period_admm.py
```

Run the synthetic walk-forward comparison:

```text
python backtest.py
```

Run the same comparison on a local market-data CSV. The file must contain a date column followed by one numeric column per asset. Use decimal returns such as `0.01` for one percent, or pass `--kind prices` for positive price levels:

```text
python backtest.py --csv returns.csv --kind returns --periods 150 --cost-bps 10 --rebalance-every 5
python backtest.py --csv prices.csv --kind prices --periods 150 --cost-bps 10
python backtest.py --csv returns.csv --kind returns --forecast-method lag1 --periods 150
```

The command does not download data. It keeps the data boundary explicit and lets the same walk-forward code run on a CSV you obtained separately.

Run all tests:

```text
python -m unittest -v
```

## Deterministic model

For each estimation window, EWMA estimates expected returns and Ledoit-Wolf estimates the covariance matrix. The optional `forecast_method="lag1"` setting fits one lagged linear model per asset using only that window. The optimizer solves a multi-period mean-variance objective with an L1 penalty on trades. Soft-thresholding creates exact zero trades, while a consensus split enforces the simplex constraint correctly.

The backtest estimates parameters only from returns before each decision. Its default penalty is the charged transaction cost:

```python
lambda_from_cost(cost_bps) == cost_bps / 10_000
```

Use `compare_costs` to repeat the same test at several cost levels:

```python
from backtest import compare_costs

print(compare_costs(returns, cost_levels=(5, 10, 20), periods=150))
```

## RL benchmark

`rl_policy.py` contains a deliberately small policy-gradient agent. Its state is the recent mean return, recent volatility, and current portfolio weights. A linear policy produces logits, softmax converts them into portfolio weights, and Gaussian logit noise provides exploration during training. The reward is portfolio return minus the same linear turnover cost used by the backtest.

The agent trains once on the historical window before the test period. It then acts deterministically during the test period. This prevents future test returns from entering training, but it is not a production RL system or a claim of predictive performance.

## Synthetic smoke result

The default optimizer run uses 50 periods and 20 assets. The corrected solver produced:

```text
iterations                281
final_primal_residual      0.000068
final_dual_residual        0.000099
zero_trade_fraction        0.897
final_active_positions    20
final_effective_assets    14.60
```

These are synthetic-data checks, not market-performance claims.

## Synthetic backtest result

Using 150 periods and a 10-basis-point cost:

```text
                       buy_and_hold  equal_weight  minimum_variance  single_period  multi_period  rl_policy
total_net_return            -0.0350       -0.0320           -0.0296        -0.0549       -0.0422    -0.1531
annualized_return           -0.0581       -0.0531           -0.0492        -0.0906       -0.0699    -0.2435
annualized_volatility        0.0426        0.0423            0.0240         0.0259        0.0257     0.1821
sharpe_net                  -1.3840       -1.2680           -2.0874        -3.6474       -2.8019    -1.4415
max_drawdown                -0.0458       -0.0440           -0.0365        -0.0570       -0.0450    -0.2224
mean_turnover                0.0000        0.0086            0.0517         0.1356        0.0740     0.2721
total_cost                   0.0000        0.0013            0.0078         0.0203        0.0111     0.0408
zero_trade_fraction          1.0000        0.0080            0.0473         0.0830        0.3710     0.8940
```

On this run, the RL benchmark underperformed the deterministic strategies and had higher volatility and turnover. It is included as a reproducible baseline for comparison, not as evidence that reinforcement learning improves portfolio allocation.

## Local CSV input

`data.py` loads a local file whose first column is a date and whose remaining columns are assets:

```python
from data import load_returns_csv
from backtest import compare

returns = load_returns_csv("returns.csv", kind="returns")
print(compare(returns, periods=150))
```

Use `kind="prices"` to convert positive price levels to simple returns. The adapter checks dates, numeric values, missing values, and price validity. The repository does not include a downloaded market dataset, so put a CSV obtained separately beside the project or pass its path to the command above. The same walk-forward code then replaces the synthetic input without changing the optimizer.


## Limitations

The included results use deterministic synthetic returns and are only an implementation check. The project does not claim profitability or real-market superiority. The cost model includes linear turnover cost only; it does not model spreads, slippage, market impact, leverage, borrowing, or shorting. The RL agent is a small benchmark, not a production trading system.

## Project files

- `multi_period_admm.py`: EWMA or lag-1 return estimates, simplex and soft-thresholding utilities, consensus ADMM solver, and deterministic smoke run.
- `backtest.py`: walk-forward evaluation, six strategy comparisons, transaction-cost accounting, rebalancing control, cost sensitivity, and the local-data CLI.
- `rl_policy.py`: NumPy REINFORCE-style policy used by the optional RL benchmark.
- `data.py`: validation and loading for local return or price CSV files.
- `test_*.py`: 72 offline tests covering the solver, backtest, CSV adapter, forecasting option, and RL path.

The final verification uses Python 3.11.9 with the pinned dependencies in `requirements.txt`. The full test suite, optimizer smoke run, and backtest script all pass.
