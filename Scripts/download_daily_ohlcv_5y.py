#project/download_daily_ohlcv_5y.py

import yfinance as yf
import pandas as pd
from pathlib import Path
import numpy as np

# -------------------------
# 1. Tickers from your list
# -------------------------
TICKERS = [
    "AMD", "COIN", "LLY",
]

# -------------------------------------------------------
# 2. Output directory (your exact specified folder)
#    Relative to repo root: project/data/OHLCV 5y/
# -------------------------------------------------------
out_dir = Path(r"project/data/OHLCV 5y")
out_dir.mkdir(parents=True, exist_ok=True)

ohlcv_parquet = out_dir / "daily_ohlcv_5y.parquet"
rets_parquet  = out_dir / "daily_returns_5y.parquet"

ohlcv_csv = out_dir / "evt_gpd_sleeve_ohlcv_5y.csv"
rets_csv  = out_dir / "evt_gpd_daily_returns_5y.csv"

# -------------------------------------
# 3. Download 5 years of DAILY OHLCV
# -------------------------------------
print("Downloading 5y daily OHLCV...")

data = yf.download(
    tickers=TICKERS,
    period="5y",
    interval="1d",
    auto_adjust=False,
    group_by="ticker",
    threads=True
)

rows = []
for ticker in TICKERS:
    if ticker not in data.columns.get_level_values(0):
        print(f"⚠️  Warning: no data returned for {ticker} — skipping.")
        continue

    df_t = data[ticker].copy()
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    missing = [c for c in expected_cols if c not in df_t.columns]
    if missing:
        print(f"⚠️  {ticker}: missing {missing}, skipping.")
        continue

    df_t = df_t[expected_cols]
    df_t["ticker"] = ticker
    df_t["date"] = df_t.index
    rows.append(df_t)

if not rows:
    raise RuntimeError("No valid OHLCV data downloaded for any ticker.")

ohlcv = pd.concat(rows, axis=0).reset_index(drop=True)
ohlcv = ohlcv.rename(columns={
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume"
})

print(f"✅ OHLCV shape: {ohlcv.shape}")

# save parquet + csv
ohlcv.to_parquet(ohlcv_parquet, index=False)
ohlcv.to_csv(ohlcv_csv, index=False)
print(f"Saved OHLCV to:\n  {ohlcv_parquet}\n  {ohlcv_csv}")

# ---------------------------------------
# 4. Compute daily log returns (Adj Close)
# ---------------------------------------
print("Computing daily log returns...")

ohlcv_sorted = ohlcv.sort_values(["ticker", "date"])

# log-return: log(P_t / P_{t-1})
ohlcv_sorted["log_ret"] = np.log(
    ohlcv_sorted.groupby("ticker")["adj_close"].shift(0) /
    ohlcv_sorted.groupby("ticker")["adj_close"].shift(1)
)

rets = ohlcv_sorted.dropna(subset=["log_ret"]).copy()

print(f"✅ Returns shape: {rets.shape}")

# save parquet + csv
rets.to_parquet(rets_parquet, index=False)
rets.to_csv(rets_csv, index=False)

print(f"Saved returns to:\n  {rets_parquet}\n  {rets_csv}")

print("\n🎉 Done! Daily OHLCV + returns saved in both Parquet and CSV formats.")
