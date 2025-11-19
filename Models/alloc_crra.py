# alloc_crra.py
# Optimal long-only mean–variance weights under Arrow–Pratt (CRRA ≈ gamma).
# Also prints 3 restricted-universe allocations:
#   - Best by Sharpe
#   - Best by Adjusted Sharpe
#   - Best by Sortino
#
# Examples:
#   python alloc_crra.py --gamma 3.0 --years 5 --freq ME
#   python alloc_crra.py --tickers NVDA AAPL MSFT SPY HYG --gamma 4
#   python alloc_crra.py --top3_report --anchors SPY HYG --freq ME --years 5

import argparse, numpy as np, pandas as pd
from pathlib import Path

# Try scipy; if missing we fall back to closed-form + simplex projection
try:
    from scipy.optimize import minimize
    _SCIPY = True
except Exception:
    _SCIPY = False

# Try yfinance for prices
try:
    import yfinance as yf
    _YF = True
except Exception:
    _YF = False


# ------------------------ math helpers ------------------------

def project_to_simplex(v):
    """Euclidean projection of v onto {w: w>=0, sum w=1}."""
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.where(u > (cssv - 1) / (np.arange(n) + 1))[0]
    theta = 0.0 if len(rho_idx) == 0 else (cssv[rho_idx[-1]] - 1) / (rho_idx[-1] + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        w[np.argmax(v)] = 1.0
    else:
        w /= s
    return w


def annualize(ret_m, cov_m, freq):
    # Monthly (ME) -> 12; weekly -> 52; daily (1d) -> 252
    if freq == "ME":
        k = 12
    elif freq in ("W", "W-FRI", "WEEKLY"):
        k = 52
    else:
        k = 252
    mu_ann  = ret_m * k
    cov_ann = cov_m * k
    return mu_ann, cov_ann


def fetch_prices(tickers, years=5, freq="ME"):
    if not _YF:
        raise RuntimeError("yfinance not available. Please install yfinance.")
    tickers = [t.upper() for t in tickers]
    per = f"{years}y"

    df = yf.download(
        tickers if len(tickers) > 1 else tickers[0],
        period=per,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Normalize to a 2D price frame 'px' with columns=tickers
    if isinstance(df.columns, pd.MultiIndex):
        lvl1 = set(df.columns.get_level_values(1))
        key = "Adj Close" if "Adj Close" in lvl1 else ("Close" if "Close" in lvl1 else None)
        if key is None:
            raise RuntimeError("Neither 'Adj Close' nor 'Close' present in yfinance frame.")
        px = df.xs(key, axis=1, level=1)
        # keep only requested tickers & drop duplicate cols
        keep = [t for t in tickers if t in px.columns]
        px = px.loc[:, keep]
    else:
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]].copy()
        elif "Close" in df.columns:
            px = df[["Close"]].copy()
        else:
            raise RuntimeError(f"No price column found. Available: {list(df.columns)}")
        px.columns = [tickers[0]]

    # Resample to requested frequency
    if freq == "ME":
        px = px.resample("ME").last()
    elif freq.upper().startswith("W"):
        px = px.resample("W-FRI").last()
    # else: daily -> leave as is

    # Clean: tolerate up to 10% missing per column, ffill then drop remaining NAs
    px = px.dropna(how="all")
    px = px.loc[:, ~px.columns.duplicated()]            # dedupe
    px = px[px.columns[px.isna().mean() <= 0.10]].ffill().dropna()

    if px.empty:
        raise RuntimeError("Price frame empty after cleaning; try more years or check tickers.")
    return px


def mv_opt_longonly(mu, Sigma, gamma=3.0, w0=None):
    """
    Solve: minimize  0.5*gamma * w'Σw - μ'w   s.t. sum w = 1, w>=0
    """
    mu = np.asarray(mu, float).reshape(-1)
    Sigma = np.asarray(Sigma, float)
    n = len(mu)

    if w0 is None:
        # closed-form (unconstrained) then project to simplex
        w0 = np.linalg.pinv(Sigma).dot(mu) / max(1e-12, gamma)
        w0 = project_to_simplex(w0)

    def obj(w):
        return 0.5 * gamma * float(w @ Sigma @ w) - float(mu @ w)

    def grad(w):
        return gamma * (Sigma @ w) - mu

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0,
             'jac': lambda w: np.ones_like(w)})
    bnds = [(0.0, 1.0) for _ in range(n)]

    if _SCIPY:
        res = minimize(obj, w0, method="SLSQP", jac=grad, bounds=bnds, constraints=[cons],
                       options=dict(maxiter=500, ftol=1e-12, disp=False))
        if res.success:
            return res.x
        # fallback to projected gradient

    # projected gradient fallback
    w = w0.copy()
    step = 1.0 / (gamma * (np.linalg.norm(Sigma, 2) + 1e-8))
    for _ in range(2000):
        w = w - step * grad(w)
        w = project_to_simplex(w)
    return w


# ------------------------ metrics helpers ------------------------

def per_period_rate(annual_rate, freq):
    k = 12 if freq == "ME" else (52 if freq in ("W", "W-FRI", "WEEKLY") else 252)
    return (1 + annual_rate) ** (1 / k) - 1, k

def portfolio_series(rets_log_df, w, names):
    """Per-period portfolio *simple* returns from per-period log returns DF, sliced to `names`."""
    rs = np.expm1(rets_log_df[names])   # simple per-period returns per asset
    rp = (rs @ np.asarray(w)).rename("rp")
    return rp

def sharpe_sortino_adjusted(rp_simple, rf_annual, target_annual, freq):
    """Compute Sharpe, Sortino, and adjusted Sharpe (Pezier–White) from simple per-period returns."""
    rf_p, k  = per_period_rate(rf_annual, freq)
    tgt_p, _ = per_period_rate(target_annual, freq)

    ex = rp_simple - rf_p
    dd = np.minimum(rp_simple - tgt_p, 0.0)

    mu_ex  = ex.mean()
    sd_ex  = ex.std(ddof=1)
    dd_vol = np.sqrt((dd**2).mean())

    sr_p    = np.nan if sd_ex == 0 else mu_ex / sd_ex
    sort_p  = np.nan if dd_vol == 0 else mu_ex / dd_vol

    sr_ann   = sr_p * np.sqrt(k)
    sort_ann = sort_p * np.sqrt(k)

    # pandas Series.kurt() returns **excess** kurtosis by default
    S  = ex.skew()
    Kx = ex.kurt()
    adj_sr_ann = sr_ann * (1 + (S/6.0)*sr_ann - (Kx/24.0)*(sr_ann**2))
    return sr_ann, sort_ann, adj_sr_ann


def summarize(weights, mu_ann, Sigma_ann, names, rets_log_df, rf_annual, target_annual, freq):
    names = list(names)  # ensure indexable sequence
    w   = np.asarray(weights)
    er  = float(np.asarray(mu_ann) @ w)
    vol = float(np.sqrt(w @ np.asarray(Sigma_ann) @ w))
    rp  = portfolio_series(rets_log_df, w, names)
    sr, sortino, adj_sr = sharpe_sortino_adjusted(rp, rf_annual, target_annual, freq)

    mu_series = pd.Series(np.asarray(mu_ann), index=names)
    tab = pd.DataFrame({"asset": names, "weight": w, "ann_mu": mu_series.values})
    return tab, er, vol, sr, sortino, adj_sr


# --------- per-asset ranking for Top-3 selection (+anchors) ---------

def rank_equities_by_metric(rets_log_df, anchors, metric, rf_annual, target_annual, freq):
    """
    Return list of equities (non-anchors) ranked by chosen metric in descending order.
    metric in {"sharpe","adjsharpe","sortino"}.
    """
    anchors = set(anchors)
    cols = [c for c in rets_log_df.columns if c not in anchors]
    if not cols:
        return []

    scores = {}
    for c in cols:
        sr, sortino, adj = sharpe_sortino_adjusted(np.expm1(rets_log_df[c]), rf_annual, target_annual, freq)
        if metric == "sharpe":
            scores[c] = sr
        elif metric == "adjsharpe":
            scores[c] = adj
        else:
            scores[c] = sortino
    ranked = pd.Series(scores).sort_values(ascending=False).index.tolist()
    return ranked


# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["NVDA","AAPL","MSFT","SPY","HYG"])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--freq", default="ME", choices=["ME","W-FRI","1d"])
    ap.add_argument("--gamma", type=float, default=3.0, help="Arrow–Pratt risk aversion (higher = more bonds)")
    ap.add_argument("--gamma_sweep", nargs="*", type=float, default=[2.0,3.0,4.0,6.0])
    ap.add_argument("--outdir", default="project/data/alloc", type=str)
    ap.add_argument("--rf", type=float, default=0.02, help="Annual risk-free rate (e.g., 0.02 = 2%)")
    ap.add_argument("--target", type=float, default=0.0, help="Annual target for Sortino (0 = MAR)")
    ap.add_argument("--anchors", nargs="*", default=["SPY","HYG"], help="Tickers to always include if present.")
    ap.add_argument("--top3_report", action="store_true",
                    help="If set, print three allocations for Top-3 equities (by Sharpe / AdjSharpe / Sortino) + anchors.")
    args = ap.parse_args()

    tickers = [t.upper() for t in args.tickers]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Prices -> per-period log returns
    px = fetch_prices(tickers, years=args.years, freq=args.freq)
    survivors = px.columns.tolist()
    if set(survivors) != set(tickers):
        dropped = sorted(set(tickers) - set(survivors))
        if dropped:
            print("[warn] dropped (missing/NA):", dropped)
    tickers = survivors  # keep exact cleaned order/columns

    rets = np.log(px / px.shift(1)).dropna()
    mu_m  = rets[tickers].mean()
    cov_m = rets[tickers].cov()

    # 2) Annualize (full universe)
    mu_full, Sigma_full = annualize(mu_m.values, cov_m.values, args.freq)

    # 3) Solve for requested gamma on full universe (informative baseline)
    w_star_full = mv_opt_longonly(mu_full, Sigma_full, gamma=args.gamma)
    tab, er, vol, sr, sortino, adj_sr = summarize(
        w_star_full, mu_full, Sigma_full, tickers, rets, args.rf, args.target, args.freq
    )
    tab.to_csv(outdir / f"weights_gamma_{args.gamma:.2f}.csv", index=False)

    print("\n=== Optimal Weights (full universe, long-only) ===")
    print(tab.to_string(index=False,
          formatters={"weight": "{:.4f}".format, "ann_mu": "{:.3%}".format}))
    print(f"\nPortfolio ER: {er:.2%}  Vol: {vol:.2%}")
    print(f"Sharpe: {sr:.2f} | Sortino: {sortino:.2f} | Adjusted Sharpe: {adj_sr:.2f}")

    # 4) Gamma sweep (full universe)
    rows = []
    for g in args.gamma_sweep:
        w = mv_opt_longonly(mu_full, Sigma_full, gamma=g)
        _, er_g, vol_g, sr_g, sort_g, adj_g = summarize(
            w, mu_full, Sigma_full, tickers, rets, args.rf, args.target, args.freq
        )
        row = {"gamma": g, "ER": er_g, "Vol": vol_g,
               "Sharpe": sr_g, "Sortino": sort_g, "AdjSharpe": adj_g}
        row.update({f"w_{t}": wi for t, wi in zip(tickers, w)})
        rows.append(row)
    sweep = pd.DataFrame(rows).sort_values("gamma")
    sweep.to_csv(outdir / "weight_sweep.csv", index=False)

    print("\n=== Gamma sweep (full universe) ===")
    cols = ["gamma","ER","Vol","Sharpe","Sortino","AdjSharpe"] + [f"w_{t}" for t in tickers]
    fmts = {"ER":"{:.2%}".format,"Vol":"{:.2%}".format,
            "Sharpe":"{:.2f}".format,"Sortino":"{:.2f}".format,"AdjSharpe":"{:.2f}".format}
    fmts.update({f"w_{t}":"{:.3f}".format for t in tickers})
    print(sweep[cols].to_string(index=False, formatters=fmts))

    # 5) Correlation table (full universe)
    corr = rets[tickers].corr()
    corr.to_csv(outdir / "corr_table.csv")
    print("\nSaved:",
          outdir / f"weights_gamma_{args.gamma:.2f}.csv", "|",
          outdir / "weight_sweep.csv", "|",
          outdir / "corr_table.csv")

    # 6) Top-3 report: (top 3 equities by metric) + anchors -> optimize & print three allocations
    if args.top3_report:
        anchors = [a for a in args.anchors if a in tickers]
        if anchors:
            print(f"\n[info] anchors present: {anchors}")
        else:
            print("\n[info] no anchors present (after cleaning).")

        def solve_and_print(title, ranked_equities):
            top3 = [e for e in ranked_equities if e in tickers][:3]
            universe = anchors + top3
            # ensure unique order
            seen = set()
            universe = [x for x in universe if not (x in seen or seen.add(x))]
            if len(universe) < 2:
                print(f"\n{title}: insufficient assets after filtering. Skipping.")
                return
            # Rebuild stats on restricted universe
            mu_m_sub  = rets[universe].mean()
            cov_m_sub = rets[universe].cov()
            mu_sub, Sigma_sub = annualize(mu_m_sub.values, cov_m_sub.values, args.freq)
            w = mv_opt_longonly(mu_sub, Sigma_sub, gamma=args.gamma)
            tab_sub, er_sub, vol_sub, sr_sub, sort_sub, adj_sub = summarize(
                w, mu_sub, Sigma_sub, universe, rets, args.rf, args.target, args.freq
            )
            # Pretty print
            print(f"\n=== {title} (Top-3 + anchors, gamma={args.gamma:.2f}) ===")
            print(f"Members: {universe}")
            print(f"Sharpe: {sr_sub:.2f} | AdjSharpe: {adj_sub:.2f} | Sortino: {sort_sub:.2f}")
            alloc_line = ", ".join([f"{a} {wt:.1%}" for a, wt in zip(tab_sub['asset'], tab_sub['weight'])])
            print(alloc_line)

        # Rank and print three ways
        ranked_sharpe   = rank_equities_by_metric(rets[tickers], anchors, "sharpe",    args.rf, args.target, args.freq)
        ranked_adj      = rank_equities_by_metric(rets[tickers], anchors, "adjsharpe", args.rf, args.target, args.freq)
        ranked_sortino  = rank_equities_by_metric(rets[tickers], anchors, "sortino",   args.rf, args.target, args.freq)

        solve_and_print("Best by Sharpe",          ranked_sharpe)
        solve_and_print("Best by Adjusted Sharpe", ranked_adj)
        solve_and_print("Best by Sortino",         ranked_sortino)


if __name__ == "__main__":
    main()