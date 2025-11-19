#make_crra_figs_longsafe.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

inpath = Path("project/data/alloc_longsafe/weight_sweep.csv")
outdir = Path("figures"); outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inpath)

gammas = df["gamma"].values
mu = df["ER"].values      # already in decimal form (e.g. 0.424)
vol = df["Vol"].values

# ---------- Efficient frontier (μ vs σ) for long-safe ----------
plt.figure(figsize=(8,5))
plt.plot(vol, mu, marker="o")
for x, y, g in zip(vol, mu, gammas):
    plt.annotate(f"γ={int(g)}", (x, y), textcoords="offset points", xytext=(6,6))
plt.title("Efficient Frontier — Long-Safe Portfolio (CRRA Sweep)")
plt.xlabel("Volatility (σ)")
plt.ylabel("Expected Return (μ)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "longsafe_efficient_frontier.png", dpi=200)
plt.close()

# ---------- Weights vs γ for the 5 long-safe assets ----------
assets = ["MCK", "AVGO", "PWR", "SPY", "HYG"]

plt.figure(figsize=(9,5))
for a in assets:
    plt.plot(gammas, df[f"w_{a}"].values, marker="o", label=a)
plt.title("Optimal Weights vs. Risk Aversion — Long-Safe Portfolio")
plt.xlabel("Risk Aversion (γ)")
plt.ylabel("Portfolio Weight")
plt.ylim(0, 1.05)
plt.legend(title="Assets", loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "longsafe_weights_vs_gamma.png", dpi=200)
plt.close()

print("Saved long-safe figures.")
