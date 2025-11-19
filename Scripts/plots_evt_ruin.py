# project/plots_evt_ruin.py
#
# EVT-based visualizations:
#   1) 3D ruin probability surface for COIN
#        + ruin ridge R = 1/L
#        + post–4.5% move path
#        + translucent "safety band" between them
#   2) 3D sigma–xi–ruin scatter (COIN, AMD, LLY)
#   3) 2D tail severity vs frequency scatter
#   4) 2D ruin probability curves vs leverage
#        + markers at L = 1, 2, 3 for COIN
#
# Inputs:
#   C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\data\evt_results\evt_gpd_params.csv
#   C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\data\evt_results\ruin_probs_single_name.csv  (optional)

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_EVT = Path(r"C:/Users/hocke/Desktop/quant_portfolio_scaffold/project/data/evt_results")
PARAMS_FILE = BASE_EVT / "evt_gpd_params.csv"

FIG_DIR = Path(r"C:/Users/hocke/Desktop/quant_portfolio_scaffold/project/figures/evt")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# EVT helper functions (same logic as in evt_ruin_model_gpd.py)
# ---------------------------------------------------------------------

def tail_prob(raw_loss_level, sigma, u, xi, beta, p_u):
    """
    P(daily loss > raw_loss_level) under vol-normalized EVT.

    raw_loss_level : e.g. 0.05 for 5% loss
    sigma          : daily log-return stdev
    u              : POT threshold in sigma units
    xi, beta       : GPD params for exceedances (shape, scale)
    p_u            : tail frequency P(X>u)
    """
    x_star = raw_loss_level / sigma

    if x_star <= u:
        return np.nan

    y = x_star - u
    cond_tail = (1.0 + xi * y / beta) ** (-1.0 / xi)
    return p_u * cond_tail


def ruin_prob_for_leverage(L, sigma, u, xi, beta, p_u):
    """Ruin if r < -1/L."""
    R_ruin = 1.0 / L
    return tail_prob(R_ruin, sigma, u, xi, beta, p_u)

# ---------------------------------------------------------------------
# Load EVT parameters
# ---------------------------------------------------------------------

def load_evt_params():
    df = pd.read_csv(PARAMS_FILE)
    required_cols = {"ticker", "sigma_daily", "z_threshold_sigma", "xi", "beta", "tail_freq"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in evt_gpd_params.csv: {missing}")
    return df

# ---------------------------------------------------------------------
# Figure 1: 3D Ruin Probability Surface for COIN + safety band
# ---------------------------------------------------------------------

def plot_ruin_surface_for_coin(evt_df):
    row = evt_df[evt_df["ticker"] == "COIN"].iloc[0]
    sigma = row["sigma_daily"]
    u = row["z_threshold_sigma"]
    xi = row["xi"]
    beta = row["beta"]
    p_u = row["tail_freq"]

    # Grid for leverage and loss thresholds
    L_vals = np.linspace(1.0, 10.0, 60)          # 1x to 10x leverage
    R_vals = np.linspace(0.05, 0.20, 60)         # 5% to 20% daily loss
    L_grid, R_grid = np.meshgrid(L_vals, R_vals)

    # Surface: probability of loss > R
    P_grid = np.zeros_like(L_grid)
    for i in range(R_grid.shape[0]):
        for j in range(R_grid.shape[1]):
            R = R_grid[i, j]
            P_grid[i, j] = tail_prob(R, sigma, u, xi, beta, p_u)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    #change camera angle
    ax.view_init(elev=28, azim=-25)

    # Loss surface
    ax.plot_surface(L_grid, R_grid * 100.0, P_grid, rstride=2, cstride=2, alpha=0.7)

    # Ruin ridge: R = 1/L
    L_ridge = np.linspace(1.0, 10.0, 200)
    R_ridge = 1.0 / L_ridge
    P_ridge = np.array([tail_prob(r, sigma, u, xi, beta, p_u) for r in R_ridge])

    ax.plot(L_ridge, R_ridge * 100.0, P_ridge, linewidth=2, label="Ruin ridge: R = 1/L")

    # Post-move path: after COIN drops 4.50833% in our favor
    move = 0.0450833
    R_post = np.clip(1.0 / L_ridge - move, 0.001, None)   # new loss threshold to wipe equity
    P_post = np.array([tail_prob(r, sigma, u, xi, beta, p_u) for r in R_post])

    ax.plot(L_ridge, R_post * 100.0, P_post,
            linestyle="--", linewidth=2, label="After -4.5% move")

    # === Safety band between ruin ridge and post-move path ===
    # Build a single closed polygon in (L, R, P) space connecting:
    #   ridge (forward in L) and post-move path (backward in L).
    verts = []
    lift = 0.006  # small vertical offset so band sits just above surface

    # forward along ridge
    for L, R, P in zip(L_ridge, R_ridge * 100.0, P_ridge):
        verts.append((L, R, P + lift))
    # back along post-move path
    for L, R, P in zip(L_ridge[::-1], R_post[::-1] * 100.0, P_post[::-1]):
        verts.append((L, R, P + lift))

    band = Poly3DCollection([verts], alpha=0.25)
    ax.add_collection3d(band)

    # Mark our specific leverage points on post-move path (L = 1,2,3)
    for L_mark in [1.0, 2.0, 3.0]:
        R_m = max(1.0 / L_mark - move, 0.001)
        P_m = tail_prob(R_m, sigma, u, xi, beta, p_u)
        ax.scatter(L_mark, R_m * 100.0, P_m)
        ax.text(L_mark, R_m * 100.0, P_m, f"L={L_mark:.0f}", fontsize=8)

    ax.set_xlabel("Leverage L")
    ax.set_ylabel("Loss threshold R (%)")
    ax.set_zlabel("P(loss > R)")
    ax.set_title("COIN: EVT Loss Tail, Ruin Ridge & Post-Move Path")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "coin_ruin_surface_3d.png", dpi=300)

# ---------------------------------------------------------------------
# Figure 2: 3D sigma–xi–ruin scatter (L = 10)
# ---------------------------------------------------------------------

def plot_sigma_xi_ruin_scatter(evt_df, L=10.0):
    records = []
    for _, row in evt_df.iterrows():
        sigma = row["sigma_daily"]
        u = row["z_threshold_sigma"]
        xi = row["xi"]
        beta = row["beta"]
        p_u = row["tail_freq"]
        p_ruin = ruin_prob_for_leverage(L, sigma, u, xi, beta, p_u)
        records.append(
            {
                "ticker": row["ticker"],
                "sigma": sigma,
                "xi": xi,
                "p_ruin": p_ruin,
            }
        )

    df_r = pd.DataFrame(records)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(df_r["sigma"], df_r["xi"], df_r["p_ruin"])
    for _, r in df_r.iterrows():
        ax.text(r["sigma"], r["xi"], r["p_ruin"], r["ticker"])

    ax.set_xlabel("Daily vol σ")
    ax.set_ylabel("Tail index ξ")
    ax.set_zlabel(f"P(ruin in 1 day | L={L:.0f})")
    ax.set_title("σ–ξ–Ruin Scatter (EVT, L=10)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "sigma_xi_ruin_3d.png", dpi=300)

# ---------------------------------------------------------------------
# Figure 3: Tail severity vs frequency scatter (2D)
# ---------------------------------------------------------------------

def plot_tail_severity_vs_frequency(evt_df):
    fig, ax = plt.subplots()

    x = evt_df["xi"]          # tail severity
    y = evt_df["tail_freq"]   # fraction of days in tail region

    ax.scatter(x, y)
    for _, row in evt_df.iterrows():
        ax.text(row["xi"], row["tail_freq"], row["ticker"])

    ax.set_xlabel("Tail index ξ (severity)")
    ax.set_ylabel("Tail frequency p_u")
    ax.set_title("Tail Severity vs Frequency (EVT GPD params)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "tail_severity_vs_frequency.png", dpi=300)

# ---------------------------------------------------------------------
# Figure 4: Ruin probability curves vs leverage (2D) + markers
# ---------------------------------------------------------------------

def plot_ruin_curves(evt_df, L_min=1.0, L_max=10.0, n_points=200):
    L_vals = np.linspace(L_min, L_max, n_points)

    fig, ax = plt.subplots()

    # Store the COIN curve so we can mark L=1,2,3 on it
    coin_curve = None

    for _, row in evt_df.iterrows():
        ticker = row["ticker"]
        sigma = row["sigma_daily"]
        u = row["z_threshold_sigma"]
        xi = row["xi"]
        beta = row["beta"]
        p_u = row["tail_freq"]

        p_vals = np.array([ruin_prob_for_leverage(L, sigma, u, xi, beta, p_u) for L in L_vals])
        line, = ax.plot(L_vals, p_vals, label=ticker)

        if ticker == "COIN":
            coin_curve = (L_vals, p_vals, line)

    # Mark our three short-leverage scenarios on the COIN curve
    if coin_curve is not None:
        L_vals, p_vals, _ = coin_curve
        for L_mark, label_text in zip([1.0, 2.0, 3.0], ["1× short", "2× short", "3× short"]):
            p_m = ruin_prob_for_leverage(
                L_mark,
                evt_df.loc[evt_df["ticker"] == "COIN", "sigma_daily"].iloc[0],
                evt_df.loc[evt_df["ticker"] == "COIN", "z_threshold_sigma"].iloc[0],
                evt_df.loc[evt_df["ticker"] == "COIN", "xi"].iloc[0],
                evt_df.loc[evt_df["ticker"] == "COIN", "beta"].iloc[0],
                evt_df.loc[evt_df["ticker"] == "COIN", "tail_freq"].iloc[0],
            )
            ax.scatter([L_mark], [p_m])
            ax.text(L_mark, p_m, label_text, fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("Leverage L")
    ax.set_ylabel("P(ruin in 1 day)")
    ax.set_title("Ruin Probability vs Leverage (EVT)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ruin_probability_curves.png", dpi=300)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    evt_df = load_evt_params()

    print("Loaded EVT params:")
    print(evt_df)

    plot_ruin_surface_for_coin(evt_df)
    plot_sigma_xi_ruin_scatter(evt_df)
    plot_tail_severity_vs_frequency(evt_df)
    plot_ruin_curves(evt_df)

    print(f"\nSaved figures to: {FIG_DIR}")

if __name__ == "__main__":
    main()