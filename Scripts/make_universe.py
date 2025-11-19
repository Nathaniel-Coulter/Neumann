import pandas as pd
from pathlib import Path

sp = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\sp500_constituents.csv"
ndx = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\nasdaq100_constituents.csv"
out = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\universe.csv"

def load_symbols(path):
    df = pd.read_csv(path)
    candidates = [c for c in ["symbol","Symbol","SYMBOL","ticker","Ticker","TICKER"] if c in df.columns]
    col = candidates[0] if candidates else df.columns[0]
    s = (df[col]
         .astype(str)
         .str.strip()
         .str.upper())
    s = s[s.ne("").fillna(False)]
    return s

sp_syms  = load_symbols(sp)
ndx_syms = load_symbols(ndx)

all_syms = pd.Series(pd.unique(pd.concat([sp_syms, ndx_syms]))).sort_values()
pd.DataFrame({"symbol": all_syms}).to_csv(out, index=False)

print(f"Wrote {len(all_syms)} tickers to {out}")
