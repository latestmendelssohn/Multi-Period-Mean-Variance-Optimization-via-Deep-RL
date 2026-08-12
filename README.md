# Multi-period mean-variance optimization

A small portfolio-optimization project. It compares deterministic Markowitz and multi-period ADMM allocation with a lightweight policy-gradient reinforcement-learning benchmark.

## Summary

### Question

Does a multi-period mean-variance strategy with turnover costs improve on standard portfolio methods on real data, and does a simple reinforcement learning policy improve on the deterministic strategy? Both questions are asked under the same walk-forward, transaction-cost, and constraint assumptions.

### Setup

- Universe: six adjusted-close price series from the public `taiwaich/stocks` sample, aligned on 419 shared trading days from 2013-11-07 through 2015-07-09.
- Test window: the last 150 days, walk-forward.
- Estimation window: 60 trailing days per decision.
- Cost model: 10 bps on traded notional, linear.
- Constraint: long only, fully invested. Weights sum to one and are non-negative.
- Rebalancing: daily by default. Cost and rebalancing sensitivities are covered in the parameter study.

### Methods

Six strategies compared on the same windows:

- Buy-and-hold from equal weight.
- Equal weight rebalanced each period.
- Minimum variance under Ledoit-Wolf covariance.
- Single-period mean-variance with EWMA returns and Ledoit-Wolf covariance.
- Multi-period mean-variance with an L1 turnover penalty, solved by consensus ADMM.
- A linear Monte Carlo REINFORCE policy trained on returns strictly before the test window.

Three expected-return estimators for the mean-variance family: EWMA, lag-1 OLS, and Ridge with five lags.

### Results

At 10 bps and a 60-day lookback over the 150-day window, only buy-and-hold (0.1056) and equal weight (0.0723) produced positive net returns. Minimum variance, single-period, and multi-period landed between -0.039 and -0.029: they pulled annualized volatility from about 0.188 down to 0.155 but underperformed after costs. The RL policy returned +0.0542 with the lowest turnover (0.0111) but the highest volatility (0.2331) and the deepest drawdown (-0.0941).

Among expected-return estimators for multi-period ADMM: EWMA (-0.0293) beat Ridge (-0.0401), which beat lag-1 (-0.0537). Ridge turnover (0.037) was close to EWMA (0.035) and much lower than lag-1 (0.094), consistent with L2 regularization shrinking noisy signals.

Cost sensitivity for multi-period: 5 bps gave -0.0266, 10 bps gave -0.0293, and 20 bps gave -0.0341. Rebalancing every 21 periods reduced turnover and pushed multi-period net return to -0.0006 on this sample.

### Limitations

One 150-day window on one small universe. Six assets, so sector or group constraints have no room to matter. Adjusted-close prices from a single public source with no explicit license. Linear cost only; no spreads, slippage, market impact, borrowing, or shorting. The RL benchmark is a small deliberate baseline, not a competitive method.

### What we did not find

- Evidence that any of the volatility-controlled strategies would beat buy-and-hold or equal weight outside this window.
- Evidence that any of the three expected-return estimators generalizes. EWMA led on this window, Ridge was second, lag-1 was third; a different sample could reorder them.
- Evidence that the linear RL policy is better than the deterministic strategies at a similar level of risk. It returned more but at higher volatility and a deeper drawdown, so it took a different risk-return trade-off rather than a strictly better one.

## What is included

- EWMA expected returns, an optional lag-1 linear forecast, an optional Ridge multi-lag forecast, and Ledoit-Wolf covariance estimation.
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
python backtest.py --csv returns.csv --kind returns --forecast-method ridge --periods 150
```

The `backtest.py` command does not download data. To fetch the small public sample used by this project and run it:

```text
python download_market_data.py
python backtest.py --csv data/market_prices.csv --kind prices --periods 150 --cost-bps 10
python parameter_study.py
```

The downloader uses only the Python standard library. It writes an ignored local CSV and does not run during a backtest.

Run all tests:

```text
python -m unittest -v
```

## Deterministic model

For each estimation window, EWMA estimates expected returns and Ledoit-Wolf estimates the covariance matrix. The optional `forecast_method="lag1"` setting fits one lagged linear model per asset using only that window. The optional `forecast_method="ridge"` setting fits a per-asset Ridge regression with `RIDGE_LAGS` previous returns as features and `RIDGE_ALPHA` L2 regularization. The optimizer solves a multi-period mean-variance objective with an L1 penalty on trades. Soft-thresholding creates exact zero trades, while a consensus split enforces the simplex constraint correctly.

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

Training is Monte Carlo REINFORCE. Each epoch collects one full episode, computes per-step returns-to-go with an optional discount, subtracts the episode mean and divides by the episode standard deviation to normalize advantages, and then applies one batch gradient update. The `discount` argument defaults to 1.0.

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
total_net_return            -0.0350       -0.0320           -0.0296        -0.0549       -0.0422    -0.0364
annualized_return           -0.0581       -0.0531           -0.0492        -0.0906       -0.0699    -0.0604
annualized_volatility        0.0426        0.0423            0.0240         0.0259        0.0257     0.1105
sharpe_net                  -1.3840       -1.2680           -2.0874        -3.6474       -2.8019    -0.5090
max_drawdown                -0.0458       -0.0440           -0.0365        -0.0570       -0.0450    -0.0911
mean_turnover                0.0000        0.0086            0.0517         0.1356        0.0740     0.0127
total_cost                   0.0000        0.0013            0.0078         0.0203        0.0111     0.0019
zero_trade_fraction          1.0000        0.0080            0.0473         0.0830        0.3710     0.9933
```

On this run, the RL benchmark had a small negative total net return, higher volatility than the deterministic strategies, and low turnover. It is included as a reproducible baseline, not evidence that reinforcement learning improves portfolio allocation.

## Local CSV input

`data.py` loads a local file whose first column is a date and whose remaining columns are assets:

```python
from data import load_returns_csv
from backtest import compare

returns = load_returns_csv("returns.csv", kind="returns")
print(compare(returns, periods=150))
```

Use `kind="prices"` to convert positive price levels to simple returns. The adapter checks dates, numeric values, missing values, and price validity. The repository does not commit raw market data. Use `download_market_data.py` for the documented public sample or pass another local CSV to the command above. The same walk-forward code then replaces the synthetic input without changing the optimizer.


## Public sample data

The documented sample comes from [taiwaich/stocks](https://github.com/taiwaich/stocks). Its six files share 419 dates from 2013-11-07 through 2015-07-09. The source README documents adjusted close values, but the repository does not display a license or a stronger redistribution note. The project therefore downloads the files only when requested and keeps the generated CSV ignored. See `data/README.md` for the source details and limitation.

## Public sample run

Using the downloaded six-asset sample, 150 test periods, a 60-day lookback, and 10 bps transaction costs:

```text
                       buy_and_hold  equal_weight  minimum_variance  single_period  multi_period  rl_policy
total_net_return              0.1056        0.0723           -0.0393        -0.0292       -0.0293     0.0542
annualized_return             0.1837        0.1244           -0.0651        -0.0486       -0.0487     0.0928
annualized_volatility         0.1884        0.1832            0.1547         0.1550        0.1551     0.2331
sharpe_net                    0.9889        0.7313           -0.3579        -0.2437       -0.2446     0.4961
max_drawdown                 -0.0824       -0.0813           -0.0799        -0.0806       -0.0805    -0.0941
mean_turnover                 0.0000        0.0098            0.0394         0.0378        0.0350     0.0111
total_cost                    0.0000        0.0015            0.0059         0.0057        0.0053     0.0017
```

This is one historical sample with a short common period. It is a reproducible data-path check, not evidence that any strategy will perform similarly elsewhere.

## Parameter study

`parameter_study.py` uses `data/market_prices.csv` by default. It prints the six-strategy baseline, cost levels of 5, 10, and 20 bps, and one-at-a-time sensitivity for:

- lookback windows of 20, 60, and 120 days
- rebalancing every 1, 5, and 21 periods
- EWMA, the lag-1 forecast, and the Ridge multi-lag forecast

The sensitivity tables focus on `multi_period` by default. Pass `--sensitivity-strategy single_period` or another strategy name to change that. On the documented sample, the multi-period strategy had total net return -0.0293 with a 60-day lookback, -0.0006 with 21-period rebalancing, -0.0537 with the lag-1 forecast, and -0.0401 with the Ridge forecast. These are one-sample comparisons, not general performance claims.

## Risk report

`risk_report(comparison)` reorders the columns of `compare()` into return, risk, risk-adjusted, and trading behaviour groups. Public sample, 150 periods, 60-day lookback, 10 bps:

```text
                       buy_and_hold  equal_weight  minimum_variance  single_period  multi_period  rl_policy
total_net_return             0.1056        0.0723           -0.0393        -0.0292       -0.0293     0.0542
annualized_return            0.1837        0.1244           -0.0651        -0.0486       -0.0487     0.0928
annualized_volatility        0.1884        0.1832            0.1547         0.1550        0.1551     0.2331
max_drawdown                -0.0824       -0.0813           -0.0799        -0.0806       -0.0805    -0.0941
sharpe_net                   0.9889        0.7313           -0.3579        -0.2437       -0.2446     0.4961
mean_turnover                0.0000        0.0098            0.0394         0.0378        0.0350     0.0111
total_cost                   0.0000        0.0015            0.0059         0.0057        0.0053     0.0017
```

Reading the numbers on this one sample: buy-and-hold and equal-weight were the only strategies with a positive net return, and the volatility-controlled variants (`minimum_variance`, `single_period`, `multi_period`) reduced volatility from 0.1884 to about 0.155 at the cost of return over the period. The RL benchmark returned +0.0542 but had the highest volatility (0.2331) and the deepest drawdown (-0.0941). Nothing here proves any strategy is better in general.

## ADMM vs RL

`admm_vs_rl(comparison)` extracts only the two rows of interest for the phase 9 comparison. Both use the same universe, the same 150-period test window, the same 60-day lookback, the same 10-bps linear cost, the same long-only simplex constraint, and the same summary metrics. The RL policy is trained once on returns strictly before the test window and acts deterministically afterward, so no test return enters training:

```text
                       multi_period  rl_policy
total_net_return            -0.0293     0.0542
annualized_return           -0.0487     0.0928
annualized_volatility        0.1551     0.2331
max_drawdown                -0.0805    -0.0941
sharpe_net                  -0.2446     0.4961
mean_turnover                0.0350     0.0111
total_cost                   0.0053     0.0017
```

On this sample the RL policy earned a higher net return with lower turnover but higher volatility and a deeper drawdown. The two strategies picked different trade-offs: multi-period ADMM contained volatility but underperformed after costs; the linear REINFORCE policy took more risk for a higher return. A single 150-period run is not enough to conclude that either approach is better outside this window.

## Limitations

The included results use deterministic synthetic returns and are only an implementation check. The project does not claim profitability or real-market superiority. The cost model includes linear turnover cost only; it does not model spreads, slippage, market impact, leverage, borrowing, or shorting. The RL agent is a small benchmark, not a production trading system.

## Project files

- `multi_period_admm.py`: EWMA, lag-1, or Ridge return estimates, simplex and soft-thresholding utilities, consensus ADMM solver, and deterministic smoke run.
- `backtest.py`: walk-forward evaluation, six strategy comparisons, transaction-cost accounting, rebalancing control, cost sensitivity, `risk_report`, `admm_vs_rl`, and the local-data CLI.
- `rl_policy.py`: NumPy REINFORCE-style policy used by the optional RL benchmark.
- `data.py`: validation and loading for local return or price CSV files.
- `download_market_data.py`: standard-library downloader and adjusted-close alignment for the documented public sample.
- `data/README.md`: source, date coverage, and usage caveat for the downloaded sample.
- `parameter_study.py`: baseline, cost, lookback, rebalancing, and forecast sensitivity tables.
- `test_*.py`: 84 offline tests covering the solver, backtest, CSV adapter, downloader helper, forecasting options, parameter study, reporting helpers, and RL path.

The final verification uses Python 3.11.9 with the pinned dependencies in `requirements.txt`. The full 84-test suite, optimizer smoke run, public-sample download, real-data backtest, and parameter study all pass.
