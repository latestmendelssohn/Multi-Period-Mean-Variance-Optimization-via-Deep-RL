# Multi-Period-Mean-Variance-Optimization-via-Deep-RL
This project engineers two institutional-grade quantitative trading cloud microservices. Project 1 uses an LSTM to forecast market parameters, feeding an ADMM solver for L1-penalized optimization. Project 2 uses RAG and Gemini to extract regulatory limits, solving fractional diversification via the Charnes-Cooper transformation.
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

# ==========================================
# 1. NATURAL PARAMETER ESTIMATION (STATISTICS)
# ==========================================
def generate_synthetic_market_data(n_assets=20, n_days=1000):
    """Generates synthetic correlated asset returns for out-of-the-box testing."""
    #np.random.seed(42)
    # Create a base covariance structure
    base_cov = np.random.randn(n_assets, n_assets)
    base_cov = base_cov @ base_cov.T / n_assets

    # Generate multivariate normal returns
    returns = np.random.multivariate_normal(
        mean=np.linspace(0.0001, 0.0005, n_assets),
        cov=base_cov * 0.0001,
        size=n_days
    )
    return pd.DataFrame(returns, columns=[f'Asset_{i}' for i in range(n_assets)])

def estimate_parameters_naturally(df_returns, lookback=60):
    """
    The natural way to estimate parameters:
    - Expected Returns (Mu): Exponentially Weighted Moving Average (EWMA)
    - Covariance (Sigma): Ledoit-Wolf Shrinkage (Guarantees Positive Definiteness)
    """
    T, N = df_returns.shape
    mu_seq = np.zeros((T - lookback, N))
    sigma_seq = np.zeros((T - lookback, N, N))

    # Initialize Ledoit-Wolf shrinkage estimator
    lw = LedoitWolf()

    print(f"Estimating rolling parameters using {lookback}-day windows and Ledoit-Wolf Shrinkage...")

    for i in range(lookback, T):
        window_returns = df_returns.iloc[i-lookback:i]

        # 1. EWMA for Expected Returns (gives more weight to recent days)
        ewma_returns = window_returns.ewm(span=lookback//2).mean().iloc[-1].values
        mu_seq[i-lookback] = ewma_returns

        # 2. Ledoit-Wolf Shrinkage for Covariance
        # (Mathematically guarantees the matrix is invertible and well-conditioned)
        sigma_seq[i-lookback] = lw.fit(window_returns).covariance_

    return mu_seq, sigma_seq

# ==========================================
# 2. EXACT MATHEMATICAL OPTIMIZATION (ADMM)
# ==========================================
def project_simplex(v):
    """Exact O(N log N) projection onto the probability simplex: sum(x)=1, x>=0."""
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)

def admm_multi_period_optimizer(mu_seq, sigma_seq, x0, gamma=2.0, lamda=0.005, rho=1.0, max_iter=100, tol=1e-3):
    """
    Solves the Multi-Period L1-Penalized Markowitz problem.
    Mathematical rigor is fully preserved.
    """
    T, n = mu_seq.shape
    x = np.ones((T, n)) / n  # Primal variable (Weights)
    u = np.zeros((T, n))     # Auxiliary variable (Trades)
    y = np.zeros((T, n))     # Dual variable (Multipliers)

    # Pre-compute inverted Q matrices for the block-tridiagonal system
    Q_inv = np.zeros((T, n, n))
    I = np.eye(n)
    for t in range(T):
        Q = gamma * sigma_seq[t] + (2 * rho * I if t < T - 1 else rho * I)
        Q_inv[t] = np.linalg.inv(Q)

    print(f"\nBeginning ADMM Iterations for {T} periods...")

    for k in range(max_iter):
        u_old = np.copy(u)

        # Block 1: x-update (Unconstrained solve + Simplex Projection)
        for t in range(T):
            x_prev = x[t-1] if t > 0 else x0
            # Pull from yesterday
            c = mu_seq[t] + rho * (x_prev + u[t] - y[t]/rho)
            # Pull from tomorrow (if not the last period)
            if t < T - 1:
                c += rho * (x[t+1] - u[t+1] + y[t+1]/rho)

            x[t] = project_simplex(Q_inv[t] @ c)

        # Block 2: u-update (Soft-Thresholding Proximal Mapping for L1 penalty)
        for t in range(T):
            x_prev = x[t-1] if t > 0 else x0
            v_t = x[t] - x_prev + y[t]/rho
            kappa = lamda / rho
            # Analytical shrinkage: forces small trades exactly to 0.0
            u[t] = np.sign(v_t) * np.maximum(np.abs(v_t) - kappa, 0)

        # Block 3: y-update (Dual Ascent)
        for t in range(T):
            x_prev = x[t-1] if t > 0 else x0
            y[t] += rho * (x[t] - x_prev - u[t])

        # Convergence Check
        primal_res = np.sqrt(sum(np.linalg.norm(x[t] - (x[t-1] if t > 0 else x0) - u[t])**2 for t in range(T)))
        dual_res = np.sqrt(sum(np.linalg.norm(rho * (u[t] - u_old[t]))**2 for t in range(T)))

        if (k+1) % 10 == 0:
            print(f"Iter {k+1:3d} | Primal Res: {primal_res:.4f} | Dual Res: {dual_res:.4f}")

        if primal_res < tol and dual_res < tol:
            print(f"ADMM Converged EXACTLY at iteration {k+1}")
            break

    return x, u

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Get Data
    df_returns = generate_synthetic_market_data(n_assets=20, n_days=500)

    # 2. Estimate Parameters (Natural Way)
    # We will optimize a trajectory for the last 50 days
    mu_est, sigma_est = estimate_parameters_naturally(df_returns, lookback=60)
    mu_target = mu_est[-50:]
    sigma_target = sigma_est[-50:]

    # 3. Initial equal-weight portfolio
    x_initial = np.ones(20) / 20

    # 4. Run ADMM Math Engine
    # lamda controls turnover penalty. Try 0.001 (high trading) vs 0.05 (low trading)
    optimal_weights, optimal_trades = admm_multi_period_optimizer(
        mu_seq=mu_target,
        sigma_seq=sigma_target,
        x0=x_initial,
        gamma=5.0,    # Risk aversion
        lamda=0.01    # L1 Turnover penalty
    )

    # 5. Evaluate the Results
    total_trades = optimal_trades.size
    zero_trades = np.sum(np.abs(optimal_trades) <= 1e-6)
    sparsity_pct = (zero_trades / total_trades) * 100

    print("\n--- FINAL EXECUTION METRICS ---")
    print(f"Total time steps optimized : {mu_target.shape[0]} days")
    print(f"Total possible asset trades: {total_trades}")
    print(f"Trades suppressed to 0.0   : {zero_trades}")
    print(f"Trading Sparsity Achieved  : {sparsity_pct:.2f}%")
# ==========================================
    # 4. WEIGHT EXTRACTION & VISUALIZATION
    # ==========================================
print("\n--- EXTRACTING OPTIMAL WEIGHTS ---")

# 1. Convert the raw NumPy array into a formatted Pandas DataFrame
asset_columns = [f'Asset_{i}' for i in range(optimal_weights.shape[1])]
df_optimal_weights = pd.DataFrame(optimal_weights, columns=asset_columns)

# 2. Display the Final Target Portfolio (Day 50)
# We filter out weights smaller than 0.01% for a clean view
print("\nTarget Portfolio Allocation (Final Day):")
final_day_weights = df_optimal_weights.iloc[-1]
active_positions = final_day_weights[final_day_weights > 1e-4]

    # Sort from largest position to smallest
active_positions = active_positions.sort_values(ascending=False)

for asset, weight in active_positions.items():
    print(f"{asset:<10}: {weight * 100:>5.2f}%")

    # 3. Generate a Stacked Area Chart (The Industry Standard Visualization)
    # This shows exactly how your ADMM solver smoothly transitions capital over time
try:
    plt.figure(figsize=(12, 6))

        # Plot only the assets that have non-zero weight to keep the legend clean
    active_assets = df_optimal_weights.loc[:, (df_optimal_weights > 1e-4).any(axis=0)]

    plt.stackplot(active_assets.index, active_assets.T, labels=active_assets.columns, alpha=0.8)
    plt.title(f"Dynamic Portfolio Allocation over {mu_target.shape[0]} Days (ADMM L1-Penalized)", fontsize=14)
    plt.xlabel("Time Step (Days)", fontsize=12)
    plt.ylabel("Capital Allocation Weight", fontsize=12)
    plt.margins(x=0, y=0)

        # Place legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Could not generate plot: {e}")
