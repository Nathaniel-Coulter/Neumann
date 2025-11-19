# evt_ruin_model_gpd.py
#
# Vol-normalized EVT (POT + GPD) for COIN, AMD, LLY.
# Uses exactly:
#   C:/Users/hocke/Desktop/quant_portfolio_scaffold/project/data/OHLCV 5y/gpd_evt_features_5y.csv
#
# Required columns in that file:
#   - ticker
#   - log_ret
#   - date or Date
#
# Outputs:
#   - project/data/evt_results/evt_gpd_params.csv
#   - project/data/evt_results/ruin_probs_single_name.csv

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import genpareto

# ---- config ----
TICKERS = ["COIN", "AMD", "LLY"]

Z_THRESHOLD_START = 2.5     # start here (sigma units)
Z_THRESHOLD_MIN   = 1.5     # don't go below this
Z_STEP            = 0.25    # step size when relaxing threshold
MIN_TAIL_POINTS   = 30      # minimum tail size for stable GPD fit

RUIN_LEVELS = [0.05, 0.10]  # 5%, 10% raw daily loss
LEVERAGE_LEVELS = [1.0, 2.0, 3.0, 10.0]  # 1x, 2x, 3x, 10x

# ---- fixed paths ----
FEATURES_FILE = Path(
    r"C:/Users/hocke/Desktop/quant_portfolio_scaffold/project/data/OHLCV 5y/gpd_evt_features_5y.csv"
)
OUT_DIR = Path(
    r"C:/Users/hocke/Desktop/quant_portfolio_scaffold/project/data/evt_results"
)


def load_features():
    """
    Load gpd_evt_features_5y.csv and normalize date column.
    Expects columns: ticker, log_ret, and date or Date.
    """
    print(f"Loading features from: {FEATURES_FILE}")
    df = pd.read_csv(FEATURES_FILE)

    # normalize date column name
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("Expected a 'Date' or 'date' column in features file.")

    if "log_ret" not in df.columns:
        raise ValueError("Expected a 'log_ret' column in features file.")
    if "ticker" not in df.columns:
        raise ValueError("Expected a 'ticker' column in features file.")

    return df


def fit_evt_for_ticker(df_ticker):
    """
    df_ticker: DataFrame with at least 'log_ret'
    Returns: dict with sigma, xi, beta, tail_freq, n_obs, and the final u used.
    Uses an adaptive z-threshold to ensure enough tail points.
    """
    r = df_ticker["log_ret"].dropna()
    if len(r) < 250:
        raise ValueError("Not enough data points for EVT fit.")

    # 1) daily vol
    sigma = r.std(ddof=1)

    # 2) z-scores
    z = r / sigma

    # 3) downside only => X = -z (positive losses)
    z_neg = z[z < 0]
    X = -z_neg

    # 4) adaptively choose u (in sigma units) so that X > u has enough points
    u = Z_THRESHOLD_START
    tail = X[X > u]

    while len(tail) < MIN_TAIL_POINTS and u > Z_THRESHOLD_MIN:
        u -= Z_STEP
        tail = X[X > u]

    if len(tail) < MIN_TAIL_POINTS:
        raise ValueError(
            f"Could not find at least {MIN_TAIL_POINTS} tail points even at u={u:.2f}σ; "
            f"only {len(tail)} points."
        )

    print(
        f"Fitting EVT for {df_ticker['ticker'].iloc[0]} with u={u:.2f}σ "
        f"and {len(tail)} tail points."
    )

    exceedances = tail - u

    # 5) fit GPD to exceedances
    c, loc, scale = genpareto.fit(exceedances, floc=0)  # xi = c, beta = scale

    # 6) fraction of days in tail region
    p_u = len(tail) / len(X)

    return {
        "sigma": sigma,
        "u": u,
        "xi": c,
        "beta": scale,
        "p_u": p_u,
        "n": len(r),
    }


def tail_prob(raw_loss_level, sigma, u, xi, beta, p_u):
    """
    raw_loss_level: e.g. 0.05 for 5% daily loss in price.
    Returns P(daily loss > raw_loss_level) under EVT model.
    """
    # r = -raw_loss_level => z = r / sigma; X = -z = raw_loss_level / sigma
    x_star = raw_loss_level / sigma

    if x_star <= u:
        # Not in EVT region; in the paper you'd say:
        # "below threshold, we use empirical / Gaussian approximations"
        return np.nan

    y = x_star - u
    # Conditional tail under GPD: P(X > x_star | X > u)
    cond_tail = (1 + xi * y / beta) ** (-1.0 / xi)
    return p_u * cond_tail


def ruin_prob_for_leverage(L, sigma, u, xi, beta, p_u):
    """
    Ruin if daily return r < -1/L (equity fully wiped).
    """
    R_ruin = 1.0 / L  # e.g., L=10 => 10% daily loss wipes equity
    return tail_prob(R_ruin, sigma, u, xi, beta, p_u)


def main():
    df = load_features()

    rows_evt = []
    rows_ruin = []

    for tic in TICKERS:
        df_t = df[df["ticker"] == tic].copy()
        if df_t.empty:
            print(f"[WARN] No data for {tic} in features file; skipping.")
            continue

        evt = fit_evt_for_ticker(df_t)
        sigma = evt["sigma"]
        u = evt["u"]
        xi = evt["xi"]
        beta = evt["beta"]
        p_u = evt["p_u"]

        rows_evt.append({
            "ticker": tic,
            "sigma_daily": sigma,
            "z_threshold_sigma": u,
            "xi": xi,
            "beta": beta,
            "tail_freq": p_u,
            "n_obs": evt["n"],
        })

        # Ruin probabilities for leverage levels
        for L in LEVERAGE_LEVELS:
            p_ruin = ruin_prob_for_leverage(L, sigma, u, xi, beta, p_u)
            rows_ruin.append({
                "ticker": tic,
                "scenario": f"L={L:.1f}",
                "leverage_L": L,
                "R_ruin_pct": 100.0 / L,
                "p_ruin_1d": p_ruin,
            })

        # Tail probabilities at fixed raw loss levels (no explicit leverage)
        for R in RUIN_LEVELS:
            p_loss = tail_prob(R, sigma, u, xi, beta, p_u)
            rows_ruin.append({
                "ticker": tic,
                "scenario": f"loss>{R*100:.0f}%",
                "leverage_L": np.nan,
                "R_ruin_pct": R * 100,
                "p_ruin_1d": p_loss,
            })

    evt_df = pd.DataFrame(rows_evt)
    ruin_df = pd.DataFrame(rows_ruin)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evt_path = OUT_DIR / "evt_gpd_params.csv"
    ruin_path = OUT_DIR / "ruin_probs_single_name.csv"

    evt_df.to_csv(evt_path, index=False)
    ruin_df.to_csv(ruin_path, index=False)

    print("\nSaved EVT params to:", evt_path)
    print("Saved ruin probabilities to:", ruin_path)


if __name__ == "__main__":
    main()