# continuous time (merton_markowitz) tangency

import numpy as np
import matplotlib.pyplot as plt

# === My Inputs (monthly) ===
# Can be replaced

mu = np.array([
    0.0129,   # SPY
    0.0154,   # MCK
    0.0339,   # AVGO
    0.0280,   # PWR
    0.0044    # HYG
])

Sigma = np.array([
    [0.0202184, 0.001242235, 0.002015983, 0.002172027, 0.000818488],
    [0.001242235, 0.005624787, 0.000679447, 0.000529035, 0.000391472],
    [0.002015983, 0.000679447, 0.020845039, 0.002087975, 0.000808684],
    [0.002172027, 0.000529035, 0.002087975, 0.020823273, 0.000713388],
    [0.000818488, 0.000391472, 0.000808684, 0.000713388, 0.000661997],
])

rf = 0.00333

n = len(mu)
ones = np.ones(n)

Sigma_inv = np.linalg.inv(Sigma)

# === Markowitz constants for closed-form frontier ===
A = ones @ Sigma_inv @ ones
B = ones @ Sigma_inv @ mu
C = mu   @ Sigma_inv @ mu
D = A * C - B**2

# === Efficient frontier (shorting allowed: where, sum w = 1) ===

mu_grid = np.linspace(mu.min() - 0.01, mu.max() + 0.01, 200)

sigma2_grid = (A * mu_grid**2 - 2 * B * mu_grid + C) / D
sigma_grid  = np.sqrt(sigma2_grid)


mask = sigma2_grid > 0
mu_grid   = mu_grid[mask]
sigma_grid = sigma_grid[mask]

# === Tangency portfolio (unconstrained) ===
# w* ∝ Σ^{-1} (μ − r_f 1)
k = Sigma_inv @ (mu - rf * ones)
w_tan = k / (ones @ k)

mu_tan = w_tan @ mu
sigma_tan = np.sqrt(w_tan @ Sigma @ w_tan)
sharpe_tan = (mu_tan - rf) / sigma_tan

print("Tangency (shorting allowed)")
print("weights:", np.round(w_tan, 3))
print("mu_tan:", mu_tan, "sigma_tan:", sigma_tan, "Sharpe:", sharpe_tan)

# === CAL through the tangency portfolio ===
sigma_cal = np.linspace(0, sigma_tan * 1.2, 50)
mu_cal    = rf + sharpe_tan * sigma_cal


plt.figure(figsize=(7,5))
plt.plot(sigma_grid, mu_grid, label="Efficient Frontier (shorting allowed)")
plt.plot(sigma_cal,  mu_cal,  'k--', label="CAL")

plt.scatter([sigma_tan], [mu_tan], color='red', zorder=5)
plt.annotate("Tangency",
             xy=(sigma_tan, mu_tan),
             xytext=(sigma_tan*1.05, mu_tan*1.05),
             arrowprops=dict(arrowstyle="->", lw=1))

plt.xlabel("Portfolio σ (monthly)")
plt.ylabel("Expected return μ (monthly)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()