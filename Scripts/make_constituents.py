# make_constituents.py
import re
import time
import requests
import pandas as pd
from pathlib import Path

OUTDIR = Path(".")  # current folder (your project/ folder)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    # You can add "Accept-Language": "en-US,en;q=0.9" if needed
}

def fetch_html(url, retries=3, sleep=1.5):
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.text
            else:
                print(f"[WARN] {url} -> {r.status_code}; retry {i+1}/{retries}")
        except Exception as e:
            print(f"[WARN] {url} -> {e}; retry {i+1}/{retries}")
        time.sleep(sleep)
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries")

def find_table_by_columns(tables, required_cols_any, required_cols_all=None):
    """
    Pick the first table whose columns contain ANY of required_cols_any
    and (if provided) ALL of required_cols_all.
    """
    def norm(cols):
        return [str(c).strip().lower() for c in cols]
    for i, df in enumerate(tables):
        cols = norm(df.columns)
        ok_any = any(any(rc in c for c in cols) for rc in required_cols_any)
        ok_all = True
        if required_cols_all:
            ok_all = all(any(rc in c for c in cols) for rc in required_cols_all)
        if ok_any and ok_all:
            return df
    return None

def normalize_yf_ticker(sym: str) -> str:
    """
    Convert tickers to Yahoo Finance style:
    - Replace '.' with '-' (e.g., BRK.B -> BRK-B)
    - Strip spaces
    """
    s = sym.strip().upper()
    s = s.replace(".", "-")
    s = re.sub(r"\s+", "", s)
    return s

def scrape_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = fetch_html(url)
    tables = pd.read_html(html)  # parse from string, avoids 403
    # Find the table with a Symbol column
    df = find_table_by_columns(
        tables,
        required_cols_any=["symbol"],
        required_cols_all=["security"]  # helps disambiguate
    )
    if df is None:
        # fallback: just take the first table and hope it's right
        df = tables[0]
    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    if "Symbol" not in df.columns:
        # Try alternate label variants
        sym_col = next((c for c in df.columns if c.lower() == "symbol"), None)
        if sym_col:
            df.rename(columns={sym_col: "Symbol"}, inplace=True)
        else:
            raise RuntimeError("Could not find 'Symbol' column in S&P 500 table")
    # Clean tickers
    df["Symbol"] = df["Symbol"].astype(str).apply(normalize_yf_ticker)
    # Basic sanity
    df = df.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])
    out = OUTDIR / "sp500_constituents.csv"
    df.to_csv(out, index=False)
    print(f"[OK] S&P 500 tickers saved -> {out} ({len(df)} rows)")
    return df

def scrape_nasdaq100():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = fetch_html(url)
    tables = pd.read_html(html)
    # The NASDAQ-100 page has multiple tables; find one with Ticker/Symbol
    df = find_table_by_columns(
        tables,
        required_cols_any=["ticker", "symbol"]
    )
    if df is None:
        # Heuristic: try later tables
        for t in tables[::-1]:
            if any(c.lower() in ["ticker", "symbol"] for c in map(str, t.columns)):
                df = t
                break
    if df is None:
        raise RuntimeError("Could not locate NASDAQ-100 constituents table")

    df.columns = [str(c).strip() for c in df.columns]
    # Standardize column name to "Ticker"
    if "Ticker" not in df.columns:
        # find a likely ticker column
        tcol = next((c for c in df.columns if c.lower() in ["ticker", "symbol"]), None)
        if tcol is None:
            raise RuntimeError("Could not find Ticker/Symbol column in NASDAQ-100 table")
        df.rename(columns={tcol: "Ticker"}, inplace=True)

    df["Ticker"] = df["Ticker"].astype(str).apply(normalize_yf_ticker)
    df = df.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"])
    out = OUTDIR / "nasdaq100_constituents.csv"
    df.to_csv(out, index=False)
    print(f"[OK] NASDAQ-100 tickers saved -> {out} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    sp = scrape_sp500()
    ndx = scrape_nasdaq100()
    # small preview
    print(sp[["Symbol"]].head())
    print(ndx[["Ticker"]].head())
