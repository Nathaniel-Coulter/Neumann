# get_monthly_adjclose_10y.py
#
# Pull monthly Adjusted Close for:
#      *any tickers* also drops NAN columns per stock if missing like COIN
#
# Date window:
#   Start: 2015-09-01 (to get Sep 2015 month-end price for Oct 2015 HPR)
#   End:   2025-10-02 (so we capture the Sep 2025 month-end bar)
#
# Output:
#   project/data/monthly_adjclose_SPY_AMD_LLY_COIN_HYG_2015-09_to_2025-09.csv
#
# Columns:
#   Date, SPY, AMD, LLY, COIN, HYG
# Values:
#   Monthly Adjusted Close from Yahoo (yfinance)

import pandas as pd
import yfinance as yf
from pathlib import Path

TICKERS = ["SPY", "MCK", "AVGO", "PWR", "HYG"]

START = "2015-09-01"   # need Sep 2015 for first Oct 2015 HPR
END   = "2025-10-02"   # end is exclusive; ensures we get Sep 2025 month-end

OUTDIR = Path("project/data")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "monthly_adjclose_SPY_AMD_LLY_COIN_HYG_2015-09_to_2025-09.csv"


def main():
    df = yf.download(
        TICKERS,
        start=START,
        end=END,
        interval="1mo",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Expect MultiIndex columns: (ticker, field)
    if isinstance(df.columns, pd.MultiIndex):
        # Pick out the 'Adj Close' slice and order columns by TICKERS list
        adj = df.xs("Adj Close", axis=1, level=1)
        # Ensure we only keep our tickers and in the intended order
        adj = adj[[t for t in TICKERS if t in adj.columns]]
    else:
        # Single-ticker edge case (shouldn't happen here, but just in case)
        if "Adj Close" not in df.columns:
            raise RuntimeError(f"No 'Adj Close' column found. Got columns: {df.columns.tolist()}")
        adj = df[["Adj Close"]].copy()
        adj.columns = [TICKERS[0]]

    # Drop rows where absolutely everything is NaN (e.g., before any ticker existed)
    adj = adj.dropna(how="all")

    # Normalize Date index to date only (no time component)
    adj.index = adj.index.normalize()

    # Reset index to have an explicit Date column
    out = adj.reset_index()
    # Make sure the first column is named exactly 'Date'
    if out.columns[0] != "Date":
        out = out.rename(columns={out.columns[0]: "Date"})

    # Reorder columns to: Date, SPY, AMD, LLY, COIN, HYG
    cols = ["Date"] + [c for c in TICKERS if c in out.columns]
    out = out[cols]

    # Save to CSV
    out.to_csv(OUTFILE, index=False)

    print("Saved monthly Adj Close panel to:")
    print("  ", OUTFILE)
    print("\nDate range in file:")
    print("  first:", out["Date"].min())
    print("  last: ", out["Date"].max())
    print("\nHead:")
    print(out.head().to_string(index=False))
    print("\nTail:")
    print(out.tail().to_string(index=False))


if __name__ == "__main__":
    main()
