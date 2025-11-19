import pandas as pd
from pathlib import Path

accr_path = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\accruals_quarterly_utf8.csv")
output_path = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\accruals_recent.csv")

# Load, strip any BOM/whitespace from headers
accr = pd.read_csv(accr_path, parse_dates=["period_end_date"])
accr.columns = accr.columns.str.replace('\ufeff','', regex=False).str.strip()

# Fallback if column came through as 'date' instead of 'period_end_date'
if "period_end_date" not in accr.columns and "date" in accr.columns:
    accr = accr.rename(columns={"date": "period_end_date"})

# Compute average of last 4 quarters per symbol
accr_recent = (
    accr.sort_values(["symbol", "period_end_date"])
        .groupby("symbol")
        .tail(4)
        .groupby("symbol")["wc_accruals_over_assets"]
        .mean()
        .rename("accruals_avg_4q")
        .reset_index()
)

accr_recent.to_csv(output_path, index=False)
print(f"Saved {len(accr_recent)} symbols to {output_path}")
