# make_crra_figs.py
# Creates 4 figures:
#  1) growth_efficient_frontier.png
#  2) growth_weights_vs_gamma.png
#  3) longsafe_weights_gamma3.png
#  4) longsafe_frontier_point.png
#
# Uses matplotlib only (no seaborn), one chart per figure, no custom colors.

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

outdir = Path("figures"); outdir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Growth Portfolio (LLY, AMD, EVGO, SPY, HYG)
# Table 19: γ, μp, σp, Sharpe, Sortino, AdjSharpe + weights
gammas = np.array([2.0, 3.0, 4.0, 6.0])
ER_pct  = np.array([46.31, 45.36, 44.89, 40.52])   # μp (%)
VOL_pct = np.array([30.60, 29.28, 28.80, 25.92])   # σp (%)

# Efficient frontier (μ vs σ)
plt.figure(figsize=(8,5))
plt.plot(VOL_pct/100.0, ER_pct/100.0, marker='o')
for x, y, g in zip(VOL_pct/100.0, ER_pct/100.0, gammas):
    plt.annotate(f"γ={int(g)}", (x, y), textcoords="offset points", xytext=(6,6))
plt.title("Efficient Frontier — Growth Portfolio (CRRA Sweep)")
plt.xlabel("Volatility (σ)")
plt.ylabel("Expected Return (μ)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "growth_efficient_frontier.png", dpi=200)
plt.close()

# Weights vs γ
weights_growth = {
    "LLY":  [0.477, 0.404, 0.367, 0.280],
    "AMD":  [0.523, 0.596, 0.633, 0.597],
    "EVGO": [0.000, 0.000, 0.000, 0.000],
    "SPY":  [0.000, 0.000, 0.000, 0.123],
    "HYG":  [0.000, 0.000, 0.000, 0.000],
}
plt.figure(figsize=(9,5))
for k, v in weights_growth.items():
    plt.plot(gammas, v, marker='o', label=k)
plt.title("Optimal Weights vs. Risk Aversion — Growth Portfolio")
plt.xlabel("Risk Aversion (γ)")
plt.ylabel("Portfolio Weight")
plt.ylim(0, 1.05)
plt.legend(title="Assets", loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "growth_weights_vs_gamma.png", dpi=200)
plt.close()

# -----------------------------
# Long-Safe Portfolio (Top-3 + anchors re-optimized at γ=3)
# Members & weights at γ=3: MCK 33.8%, AVGO 48.5%, PWR 17.7%, SPY 0%, HYG 0%
# Portfolio stats at γ=3: μp = 42.39%, σp = 21.46% (Sharpe ≈ 2.15)
longsafe_weights = {"MCK": 0.338, "AVGO": 0.485, "PWR": 0.177, "SPY": 0.0, "HYG": 0.0}

# Bar chart of weights (γ=3)
plt.figure(figsize=(7,5))
assets = list(longsafe_weights.keys())
vals   = [longsafe_weights[a] for a in assets]
plt.bar(assets, vals)
plt.title("Weights at γ=3 — Long-Safe Portfolio")
plt.ylabel("Portfolio Weight")
for i, v in enumerate(vals):
    plt.text(i, v + 0.012, f"{v:.1%}", ha='center')
plt.ylim(0, max(vals) + 0.12)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "longsafe_weights_gamma3.png", dpi=200)
plt.close()

# Single-point “frontier” marker for Long-Safe at γ=3
er_longsafe  = 0.4239  # 42.39%
vol_longsafe = 0.2146  # 21.46%
plt.figure(figsize=(7,5))
plt.scatter([vol_longsafe], [er_longsafe], marker='o')
plt.annotate("γ=3", (vol_longsafe, er_longsafe), textcoords="offset points", xytext=(6,6))
plt.title("Efficient Frontier (Single Point) — Long-Safe Portfolio")
plt.xlabel("Volatility (σ)")
plt.ylabel("Expected Return (μ)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "longsafe_frontier_point.png", dpi=200)
plt.close()

print("Saved to:", outdir.resolve())
