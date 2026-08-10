# Multi-Period-Mean-Variance-Optimization-via-Deep-RL
This project engineers two institutional-grade quantitative trading cloud microservices. Project 1 uses an LSTM to forecast market parameters, feeding an ADMM solver for L1-penalized optimization.
# Multi-Period Asset Allocation via ADMM Proximal Splitting

**Academic Title:** *Multi-Period Asset Allocation via ADMM Proximal Splitting: Enforcing Trading Sparsity with $L_1$-Turnover Penalties and Ledoit-Wolf Shrinkage*

## The Core Problem
Standard Markowitz portfolio optimization is a single-period model that ignores market friction. If run daily, it trades every single asset, every single day due to micro-fluctuations in data, bleeding capital to transaction costs. To fix this, an $L_1$-norm transaction penalty ($\lambda \|x_t - x_{t-1}\|_1$) must be added over a multi-period horizon. However, this absolute value function is non-differentiable and mathematically couples every time step together, causing standard quadratic solvers to crash or fail to find true zero-trade days.

## The Two-Stage Architecture

### Stage 1: Natural Parameter Estimation (Statistics)
Instead of relying on unstable historical sample matrices, this engine uses robust statistical estimators to generate forward-looking parameters:
* **Expected Returns ($\mu_t$):** Estimated using an Exponentially Weighted Moving Average (EWMA) to capture recent momentum.
* **Covariance ($\Sigma_t$):** Estimated using **Ledoit-Wolf Shrinkage**. This mathematically projects the sample covariance matrix toward a structured target, guaranteeing it is strictly Symmetric Positive-Definite (SPD) so the downstream optimizer never crashes during matrix inversion.

### Stage 2: Exact Mathematical Optimization (ADMM)
The predicted parameters are passed into a custom-built Alternating Direction Method of Multipliers (ADMM) calculus engine. 
* **Variable Splitting:** Introduces an auxiliary split variable for trading volume ($u_t = x_t - x_{t-1}$).
* **The Proximal Operator:** By isolating the non-differentiable $L_1$ penalty, the engine applies an analytical **Soft-Thresholding Proximal Operator**. 
* **The Result:** This operator acts as a mathematical filter, forcing small, noisy portfolio adjustments to become *exactly zero*, yielding a highly sparse, institutional-grade trading trajectory.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the engine: `python multi_period_admm.py`
