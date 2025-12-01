#Long only constraint removed

import numpy as np
import matplotlib.pyplot as plt

# Monthly expected returns (μ vector, in decimals)
mu = np.array([0.0129, 0.0154, 0.0339, 0.0280, 0.0044])  # SPY, MCK, AVGO, PWR, HYG

# Covariance matrix (monthly) – same order as our mu
Sigma = np.array([
    [0.0020184,   0.001242235, 0.002105983, 0.002172027, 0.000818488],
    [0.001242235, 0.005624787, 0.000674947, 0.001203095, 0.000391472],
    [0.002105983, 0.000674947, 0.008405039, 0.002824273, 0.000806840],
    [0.002172027, 0.001203095, 0.002824273, 0.007987072, 0.000713380],
    [0.000818488, 0.000391472, 0.000806840, 0.000713380, 0.000495087],
])

rf = 0.00333  # monthly risk-free rate from the sheet

# ---------- Markowitz closed-form frontier (shorting allowed) ----------

n = len(mu)
ones = np.ones(n)
Sigma_inv = np.linalg.inv(Sigma)

A = ones @ Sigma_inv @ ones        # 1ᵀ Σ⁻¹ 1
B = ones @ Sigma_inv @ mu          # 1ᵀ Σ⁻¹ μ
C = mu   @ Sigma_inv @ mu          # μᵀ Σ⁻¹ μ
D = A * C - B**2

def frontier(target_returns):
    """
    Given an array of target returns, return (sigmas, weights)
    for the unconstrained minimum-variance frontier.
    """
    sigmas = []
    weights_list = []

    for r in target_returns:
        # Lagrange multipliers λ, γ
        lam   = (C - B * r) / D
        gamma = (A * r - B) / D
        w = Sigma_inv @ (lam * ones + gamma * mu)
        weights_list.append(w)
        sigmas.append(np.sqrt(w @ Sigma @ w))

    return np.array(sigmas), np.vstack(weights_list)

r_min = 0.0
r_max = 0.04
target_mu = np.linspace(r_min, r_max, 300)
sigma_frontier, w_frontier = frontier(target_mu)

# ---------- Tangency portfolio (max Sharpe vs rf) ----------

excess = mu - rf * ones
w_tan_unnorm = Sigma_inv @ excess
w_tan = w_tan_unnorm / (ones @ w_tan_unnorm)   # normalize so Σw=1

mu_tan    = w_tan @ mu
sigma_tan = np.sqrt(w_tan @ Sigma @ w_tan)
sharpe_tan = (mu_tan - rf) / sigma_tan

print("Tangency weights:", w_tan)
print("Tangency μ:", mu_tan)
print("Tangency σ:", sigma_tan)
print("Tangency Sharpe:", sharpe_tan)

# ---------- Capital Allocation Line (CAL) ----------
sigma_cal = np.linspace(0, sigma_frontier.max() * 1.1, 50)
mu_cal = rf + sharpe_tan * sigma_cal

# ---------- Plot ----------
plt.figure(figsize=(7, 5))

# Unconstrained efficient frontier (bullet's upper branch)
plt.plot(sigma_frontier, target_mu, label="Efficient Frontier (shorting allowed)")

# CAL
plt.plot(sigma_cal, mu_cal, linestyle="--", label="CAL")

# Tangency portfolio point
plt.scatter([sigma_tan], [mu_tan], marker="o")
plt.annotate("Tangency", (sigma_tan, mu_tan),
             xytext=(sigma_tan * 1.05, mu_tan * 1.02),
             arrowprops=dict(arrowstyle="->", lw=0.8))

plt.xlabel("Portfolio σ (monthly)")
plt.ylabel("Expected return μ (monthly)")
plt.title("Efficient Frontier with Tangency Portfolio (shorting allowed)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()