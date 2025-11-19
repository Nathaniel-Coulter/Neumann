# short_candidates_features.py
#
# Pull 5y DAILY OHLCV from Yahoo via yfinance for the 20 short candidates
# and compute:
#   1) daily log returns (Adj Close)
#   2) daily high–low log ranges
#   3) realized volatility over the full 5y sample (per ticker)
#
# Run from your project root:
#   python short_candidates_features.py

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# Your 20 worst ML names
TICKERS = [
    "COIN", "APA", "CCL", "WBD", "HAL", "PCG", "VTRS", "DVN", "HOOD", "OXY",
    "SLB", "BAX", "XYZ", "IVZ", "PYPL", "TRGP", "FI", "PSKY", "NCLH", "MRNA",
]

OUTDIR = Path("project/data/short_features")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1) Download 5y of *daily* OHLCV
    data = yf.download(
        TICKERS,
        period="5y",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    # If yfinance returns a single-level frame, bail (means only one ticker worked)
    if not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError("Expected MultiIndex columns from yfinance; got single index.")

    rows = []
    realized_vols = {}

    for ticker in TICKERS:
        if ticker not in data.columns.get_level_values(0):
            print(f"[warn] {ticker} missing from download; skipping.")
            continue

        df = data[ticker].copy()  # columns: Open, High, Low, Close, Adj Close, Volume, etc.

        # Basic clean-up
        df = df.dropna(subset=["Adj Close", "High", "Low"])
        if df.empty:
            print(f"[warn] {ticker} has no clean data; skipping.")
            continue

        # 2) Daily log returns (Adj Close)
        df["log_ret"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

        # 3) High–low log range
        df["hl_log_range"] = np.log(df["High"] / df["Low"])

        # Realized volatility over full sample (annualized)
        # std of daily log returns * sqrt(252)
        rv = df["log_ret"].std(ddof=1) * np.sqrt(252)
        realized_vols[ticker] = rv

        # Store long format rows
        df["ticker"] = ticker
        df = df.reset_index().rename(columns={"index": "Date"})  # Date stays as index or column depending on version

        # Keep only useful columns
        rows.append(
            df[[
                "ticker", "Date",
                "Open", "High", "Low", "Close", "Adj Close", "Volume",
                "log_ret", "hl_log_range",
            ]]
        )

    if not rows:
        raise RuntimeError("No valid data for any ticker.")

    panel = pd.concat(rows, ignore_index=True).sort_values(["ticker", "Date"])

    # Save full panel of features
    panel.to_csv(OUTDIR / "short_candidates_5y_daily_features.csv", index=False)

    # Save per-ticker realized vol summary
    summary = (
        pd.Series(realized_vols, name="realized_vol_annual")
        .rename_axis("ticker")
        .reset_index()
    )
    summary.to_csv(OUTDIR / "short_candidates_realized_vol_summary.csv", index=False)

    print("\nSaved:")
    print("  ", OUTDIR / "short_candidates_5y_daily_features.csv")
    print("  ", OUTDIR / "short_candidates_realized_vol_summary.csv")
    print("\nRealized vols (annual):")
    print(summary.to_string(index=False, formatters={"realized_vol_annual": "{:.2%}".format}))


if __name__ == "__main__":
    main()
