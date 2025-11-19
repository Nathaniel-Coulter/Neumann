# alloc_growth_coin_short.py
#
# Build a "Growth + COIN short" portfolio using the same CRRA MV logic
# as alloc_crra.py, but with:
#   universe = {LLY, AMD, COIN, SPY, HYG}
#   1ᵀ w = 1
#   w_LLY, w_AMD, w_SPY, w_HYG >= 0
#   w_COIN ∈ [-s_max, 0]
#
# Example:
#   python alloc_growth_coin_short.py --gamma 3 --years 5 --freq ME --s_max 0.20

import argparse
import numpy as np
import pandas as pd

from alloc_crra import fetch_prices, annualize

# Try scipy; if missing we fall back to simple projected gradient
try:
    from scipy.optimize import minimize
    _SCIPY = True
except Exception:
    _SCIPY = False


def mv_opt_coin_short(mu, Sigma, gamma, idx_coin, s_max=0.20):
    """
    Minimize 0.5*gamma * w'Σw - μ'w
    s.t. 1'w = 1
         w_j >= 0 for j != idx_coin
         w_idx_coin ∈ [-s_max, 0]
    """
    mu = np.asarray(mu, float).reshape(-1)
    Sigma = np.asarray(Sigma, float)
    n = len(mu)

    # Initial guess: simple long-only solution (no short) on all names
    # Then set COIN to 0 so solver can move it negative.
    w0 = np.linalg.pinv(Sigma).dot(mu) / max(1e-12, gamma)
    w0 = np.maximum(w0, 0.0)
    if w0.sum() <= 0:
        w0[:] = 1.0 / n
    else:
        w0 /= w0.sum()
    w0[idx_coin] = 0.0

    def obj(w):
        return 0.5 * gamma * float(w @ Sigma @ w) - float(mu @ w)

    def grad(w):
        return gamma * (Sigma @ w) - mu

    # Bounds: COIN can be negative; others long-only
    bounds = []
    for i in range(n):
        if i == idx_coin:
            bounds.append((-s_max, 0.0))
        else:
            bounds.append((0.0, 1.0))

    cons = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
        "jac": lambda w: np.ones_like(w),
    })

    if _SCIPY:
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=[cons],
            options=dict(maxiter=1000, ftol=1e-12, disp=False),
        )
        if res.success:
            return res.x

    # Fallback: simple projected gradient with manual clipping on COIN + renorm
    w = w0.copy()
    step = 1.0 / (gamma * (np.linalg.norm(Sigma, 2) + 1e-8))
    for _ in range(3000):
        w = w - step * grad(w)

        # Enforce bounds
        for i in range(n):
            if i == idx_coin:
                w[i] = min(max(w[i], -s_max), 0.0)
            else:
                w[i] = max(w[i], 0.0)

        # Renormalize to keep 1'w = 1 while respecting COIN bounds
        w_coin = w[idx_coin]
        total_non_coin = w.sum() - w_coin
        target_non_coin_sum = 1.0 - w_coin
        if total_non_coin <= 0:
            # If everything else collapsed, put all long on the first non-COIN asset
            for i in range(n):
                if i != idx_coin:
                    w[i] = target_non_coin_sum if total_non_coin <= 0 else 0.0
                else:
                    w[i] = w_coin
        else:
            scale = target_non_coin_sum / total_non_coin
            for i in range(n):
                if i != idx_coin:
                    w[i] = max(0.0, w[i] * scale)
        # tiny numerical cleanup
        w /= w.sum()
    return w


# --- Sharpe helper (simple, annualized) ---

def per_period_rate(annual_rate, freq):
    if freq == "ME":
        k = 12
    elif freq in ("W", "W-FRI", "WEEKLY"):
        k = 52
    else:
        k = 252
    r_p = (1 + annual_rate) ** (1 / k) - 1
    return r_p, k


def sharpe_from_log_returns(rets_log_df, w, freq, rf_annual):
    """
    Compute annualized Sharpe from per-period log returns and weights.
    Uses simple returns, subtracts per-period RF, √k annualization.
    """
    # portfolio simple returns
    rs = np.expm1(rets_log_df)          # simple returns per asset
    rp = (rs @ np.asarray(w)).rename("rp")

    rf_p, k = per_period_rate(rf_annual, freq)
    ex = rp - rf_p

    mu_ex = ex.mean()
    sd_ex = ex.std(ddof=1)
    if sd_ex == 0:
        return np.nan
    sr_p = mu_ex / sd_ex
    sr_ann = sr_p * np.sqrt(k)
    return sr_ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=3.0,
                    help="Arrow–Pratt risk aversion (higher = more risk-averse)")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--freq", default="ME", choices=["ME", "W-FRI", "1d"])
    ap.add_argument("--s_max", type=float, default=0.20,
                    help="Maximum absolute short size for COIN (e.g. 0.20 = 20%%)")
    ap.add_argument("--rf", type=float, default=0.02,
                    help="Annual risk-free rate for Sharpe (e.g. 0.02 = 2%%)")
    args = ap.parse_args()

    # Universe: Growth + COIN + hedges
    tickers = ["LLY", "NVDA", "COIN", "SPY", "HYG"]

    # 1) Fetch prices and compute log returns (same style as alloc_crra)
    px = fetch_prices(tickers, years=args.years, freq=args.freq)
    survivors = px.columns.tolist()
    if set(survivors) != set(tickers):
        dropped = sorted(set(tickers) - set(survivors))
        if dropped:
            print("[warn] dropped (missing/NA):", dropped)
    tickers = survivors  # cleaned universe, consistent ordering

    if "COIN" not in tickers:
        raise RuntimeError(f"COIN not available in downloaded price data. Got: {tickers}")

    rets = np.log(px / px.shift(1)).dropna()
    mu_m = rets[tickers].mean()
    cov_m = rets[tickers].cov()

    # 2) Annualize
    mu_ann, Sigma_ann = annualize(mu_m.values, cov_m.values, args.freq)

    # 3) Solve constrained problem with COIN short
    idx_coin = tickers.index("COIN")
    w_star = mv_opt_coin_short(mu_ann, Sigma_ann, gamma=args.gamma,
                               idx_coin=idx_coin, s_max=args.s_max)

    # 4) Portfolio stats
    er = float(mu_ann @ w_star)
    vol = float(np.sqrt(w_star @ Sigma_ann @ w_star))
    sharpe = sharpe_from_log_returns(rets[tickers], w_star, args.freq, args.rf)

    # 5) Pretty print
    tab = pd.DataFrame({
        "asset": tickers,
        "weight": w_star,
        "ann_mu": mu_ann,
    })

    print("\n=== Growth + COIN short (LLY, AMD, COIN, SPY, HYG) ===")
    print(f"gamma = {args.gamma:.2f}, s_max (COIN) = {args.s_max:.2f}")
    print(tab.to_string(index=False,
          formatters={"weight": "{:.4f}".format,
                      "ann_mu": "{:.3%}".format}))
    print(f"\nPortfolio ER: {er:.2%}")
    print(f"Portfolio Vol: {vol:.2%}")
    print(f"Portfolio Sharpe (rf={args.rf:.2%}): {sharpe:.2f}")
    print(f"Net exposure (1ᵀw): {w_star.sum():.6f}")
    print(f"Gross exposure (∑|w_i|): {np.abs(w_star).sum():.3f}")
    print(f"COIN weight: {w_star[idx_coin]:.4f} (should be ≤ 0 and ≥ -s_max)")


if __name__ == "__main__":
    main()
