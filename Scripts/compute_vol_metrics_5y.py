#compute_vol_metrics_5y.py

import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------
# 1. Paths (same folder as your OHLCV CSV)
# ----------------------------------------
data_dir = Path(r"project/data/OHLCV 5y")
ohlcv_csv = data_dir / "evt_gpd_sleeve_ohlcv_5y.csv"

daily_features_csv = data_dir / "gpd_evt_features_5y.csv"
metrics_csv = data_dir / "gpd_evt_vol_metrics_5y.csv"

print(f"Loading OHLCV from: {ohlcv_csv}")
ohlcv = pd.read_csv(ohlcv_csv)

# Ensure correct dtypes
ohlcv["date"] = pd.to_datetime(ohlcv["date"])
ohlcv = ohlcv.sort_values(["ticker", "date"])

# ----------------------------------------
# 2. Compute daily log returns & ranges
# ----------------------------------------

# log-return from adj_close
ohlcv["log_ret"] = np.log(
    ohlcv.groupby("ticker")["adj_close"].shift(0) /
    ohlcv.groupby("ticker")["adj_close"].shift(1)
)

# High-low range in log space (Parkinson-style input)
ohlcv["hl_log_range"] = np.log(ohlcv["high"] / ohlcv["low"])

# Simple high–low range (absolute)
ohlcv["hl_range"] = ohlcv["high"] - ohlcv["low"]

# Drop first row per ticker where log_ret = NaN
daily = ohlcv.dropna(subset=["log_ret"]).copy()

print(f"✅ Daily features shape: {daily.shape}")
print(f"Saving daily features to: {daily_features_csv}")
daily.to_csv(daily_features_csv, index=False)

# ----------------------------------------
# 3. Helper functions for metrics
# ----------------------------------------

TRADING_DAYS = 252

def realized_vol_annual(log_rets: pd.Series) -> float:
    """Annualized realized volatility from daily log returns."""
    return log_rets.std(ddof=1) * np.sqrt(TRADING_DAYS)

def downside_dev_annual(log_rets: pd.Series) -> float:
    """Annualized downside deviation (only negative returns)."""
    neg = log_rets[log_rets < 0]
    if len(neg) == 0:
        return 0.0
    return np.sqrt((neg**2).mean()) * np.sqrt(TRADING_DAYS)

def max_drawdown_from_prices(prices: pd.Series) -> float:
    """
    Max drawdown over the sample, computed from adj_close.
    Returns a negative number (e.g., -0.62 for -62%).
    """
    cum_max = prices.cummax()
    drawdown = prices / cum_max - 1.0
    return drawdown.min()

def parkinson_vol_annual(hl_log_ranges: pd.Series) -> float:
    """
    Parkinson realized volatility estimator (annualized).
    Uses high-low log ranges.
    """
    if len(hl_log_ranges) == 0:
        return np.nan
    var_pk = (hl_log_ranges**2).sum() / (4 * len(hl_log_ranges) * np.log(2))
    return np.sqrt(var_pk) * np.sqrt(TRADING_DAYS)

def empirical_var(log_rets: pd.Series, alpha: float) -> float:
    """
    Empirical one-day VaR at level alpha (e.g., alpha=0.05 for 95% VaR).
    Returns a negative number (loss threshold).
    """
    return log_rets.quantile(alpha)

# ----------------------------------------
# 4. Aggregate per-ticker metrics
# ----------------------------------------

metrics_rows = []

for ticker, df_t in daily.groupby("ticker"):
    log_rets = df_t["log_ret"].dropna()
    hl_logs = df_t["hl_log_range"].dropna()

    # Safety check: we also need prices for drawdown
    prices = df_t.set_index("date")["adj_close"].sort_index()

    if len(log_rets) < 30:
        print(f"⚠️ {ticker}: fewer than 30 return observations, metrics may be unstable.")

    rv = realized_vol_annual(log_rets)
    dd = downside_dev_annual(log_rets)
    mdd = max_drawdown_from_prices(prices)
    pk_vol = parkinson_vol_annual(hl_logs)

    # empirical daily VaR at 95% and 99% (left-tail)
    var_95 = empirical_var(log_rets, 0.05)
    var_99 = empirical_var(log_rets, 0.01)

    metrics_rows.append({
        "ticker": ticker,
        "n_obs": len(log_rets),
        "realized_vol_annual": rv,
        "downside_dev_annual": dd,
        "max_drawdown_5y": mdd,         # e.g., -0.62 = -62% from peak
        "parkinson_vol_annual": pk_vol,
        "var_95_1d": var_95,           # negative number; cutoff for worst 5% days
        "var_99_1d": var_99            # negative number; cutoff for worst 1% days
    })

metrics = pd.DataFrame(metrics_rows).sort_values("ticker").reset_index(drop=True)

print(f"✅ Metrics table shape: {metrics.shape}")
print(f"Saving per-ticker metrics to: {metrics_csv}")
metrics.to_csv(metrics_csv, index=False)

print("\n🎉 Done! Generated:")
print(f"  - Daily features: {daily_features_csv}")
print(f"  - Per-ticker metrics: {metrics_csv}")
