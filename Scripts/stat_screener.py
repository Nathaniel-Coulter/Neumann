# stat_screener.py
import time, math, numpy as np, pandas as pd
from pathlib import Path
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import zscore

# ----------------------------
# Resolve paths relative to THIS file (…\project\stat_screener.py)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# ---- Dolthub accruals (preferred) ----
ACC_CSV = BASE_DIR / "accruals_recent.csv"  # columns: symbol, accruals_avg_4q
acc = pd.read_csv(ACC_CSV)

# canonicalize tickers: yfinance uses '-' for share classes; some sources use '.'
def canon(t):
    return str(t).strip().upper().replace('.', '-')

acc["symbol"] = acc["symbol"].map(canon)
acc = acc.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
acc_map = acc.set_index("symbol")["accruals_avg_4q"]

# ----------------------------
# Load constituent lists saved by make_constituents.py
# ----------------------------
sp500_csv = BASE_DIR / "sp500_constituents.csv"
nas100_csv = BASE_DIR / "nasdaq100_constituents.csv"

sp500 = pd.read_csv(sp500_csv)
nas100 = pd.read_csv(nas100_csv)

sp_col  = "Symbol" if "Symbol" in sp500.columns else sp500.columns[0]
ndx_col = "Ticker" if "Ticker" in nas100.columns else ("Symbol" if "Symbol" in nas100.columns else nas100.columns[0])

tickers_raw = sorted(set(sp500[sp_col].astype(str).tolist() + nas100[ndx_col].astype(str).tolist()))
tickers = sorted({ canon(t) for t in tickers_raw if isinstance(t, str) and t.strip() })
print(f"[INFO] Universe size: {len(tickers)} tickers (canon)")

# ----------------------------
# 2) Prices (10y daily -> monthly)
# ----------------------------
def get_prices(tks):
    # Request non–auto-adjusted so "Adj Close" exists reliably
    px = yf.download(" ".join(tks), period="10y", auto_adjust=False,
                     threads=True, progress=False, group_by='ticker')
    if isinstance(px.columns, pd.MultiIndex):
        lvl1 = set(px.columns.get_level_values(1))
        key = 'Adj Close' if 'Adj Close' in lvl1 else 'Close'
        out = px.xs(key, axis=1, level=1)
    else:
        out = px.get('Adj Close', px.get('Close', px))
    # drop all-null columns (dead symbols)
    out = out.loc[:, out.notna().any()]
    return out

BATCH = 35
batches = [tickers[i:i+BATCH] for i in range(0, len(tickers), BATCH)]
px_list = []
for ch in batches:
    try:
        p = get_prices(ch)
        px_list.append(p)
        time.sleep(1.5)  # gentle rate-limit
    except Exception as e:
        print("[WARN] batch fail", ch[:3], e)

if not px_list:
    raise RuntimeError("No price data pulled. Check internet/yfinance.")

prices = pd.concat(px_list, axis=1).sort_index()
prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

# monthly prices & returns (use month-end "ME")
mpx  = prices.resample("ME").last()
mret = mpx.pct_change()

# --- Fill EP via Dolt TTM EPS (fallback) ---
EPS_CSV = BASE_DIR / "ttm_eps.csv"
if EPS_CSV.exists() and not mpx.empty:
    try:
        eps = pd.read_csv(EPS_CSV)
        eps["symbol"] = eps["symbol"].map(canon)
        eps = eps.dropna(subset=["symbol", "ttm_eps"])
        ttm_eps = eps.set_index("symbol")["ttm_eps"].astype(float)

        # last month-end prices we already computed
        last_px = mpx.iloc[-1].astype(float)

        # align and compute EP_per_share = TTM_EPS / Price
        ep_from_eps = (ttm_eps.reindex(last_px.index) / last_px).replace([np.inf, -np.inf], np.nan)

        # Attach to the features frame: features index == tickers (canon)
        # If you insert here, ensure 'feat' exists; otherwise keep this just before building X
        feat["EP"] = feat["EP"].where(feat["EP"].notna(), ep_from_eps.reindex(feat.index))
        print(f"[INFO] EP filled from Dolt EPS for ~{feat['EP'].notna().sum()} tickers")
    except Exception as e:
        print("[WARN] Dolt TTM EPS fallback failed:", e)

# --- helper for single-ticker adjusted close ---
def get_adj_close_single(ticker, period="10y"):
    df = yf.download(ticker, period=period, auto_adjust=False,
                     progress=False, group_by="ticker")
    if isinstance(df.columns, pd.MultiIndex):
        lvl1 = set(df.columns.get_level_values(1))
        key = "Adj Close" if "Adj Close" in lvl1 else "Close"
        s = df.xs(key, axis=1, level=1, drop_level=True)
        if isinstance(s, pd.DataFrame) and ticker in s.columns:
            s = s[ticker]
        return s
    else:
        return df.get("Adj Close", df.get("Close"))

# SPY for risk measures (month-end)
spy     = get_adj_close_single("SPY", period="10y")
spy_mpx = spy.resample("ME").last()
spy_ret = spy_mpx.pct_change().dropna()

# Keep a dict of last month-end prices for all tickers to compute market cap robustly
LAST_PX = {}
if not mpx.empty:
    LAST_PX = mpx.iloc[-1].dropna().to_dict()

# ----------------------------
# 3) Fundamentals helpers
# ----------------------------
def safe_q(t, attr):
    """Return quarterly dataframe with datetime columns, or None."""
    try:
        df = getattr(yf.Ticker(t), attr)
        if df is None or df.empty:
            return None
        df.columns = pd.to_datetime(df.columns)
        return df
    except Exception:
        return None

def collect_quarterly(t):
    f = safe_q(t, "quarterly_financials")
    c = safe_q(t, "quarterly_cashflow")
    b = safe_q(t, "quarterly_balance_sheet")
    return f, c, b

def ttm_sum(df, field):
    """TTM sum of last 4 quarters for a field; NaN if unavailable."""
    if df is None or field not in df.index:
        return np.nan
    s = df.loc[field].sort_index().tail(4).sum()
    return float(s) if pd.notnull(s) else np.nan

def ttm_first(df, candidates):
    """Return TTM for the first field name that exists in df."""
    for field in candidates:
        v = ttm_sum(df, field)
        if pd.notnull(v):
            return v
    return np.nan

def q_last(df, field, k=4):
    """Last k quarterly points for a field (Series) or NaN."""
    if df is None or field not in df.index:
        return np.nan
    return df.loc[field].sort_index().tail(k)

def get_shares_outstanding(ti: yf.Ticker):
    """Robust shares: fast_info -> info -> full history"""
    try:
        s = ti.fast_info.get("shares")
        if s and s > 0:
            return float(s)
    except Exception:
        pass
    try:
        s = ti.info.get("sharesOutstanding")
        if s and s > 0:
            return float(s)
    except Exception:
        pass
    try:
        sh = ti.get_shares_full(start=None, end=None)
        if sh is not None and len(sh) > 0:
            s = float(sh.dropna().iloc[-1])
            if s > 0:
                return s
    except Exception:
        pass
    return np.nan

# ----------------------------
# 4) Feature builder (cached)
# ----------------------------
def features_for(t):
    t = canon(t)
    fpath = RAW / f"{t}.parquet"
    if fpath.exists():
        try:
            return pd.read_parquet(fpath).iloc[0].to_dict()
        except Exception:
            pass  # rebuild if cache broken

    out = {"ticker": t}

    # -------- MARKET CAP (robust) --------
    mcap = np.nan
    try:
        ti = yf.Ticker(t)
        # price: prefer what we already downloaded
        price = LAST_PX.get(t, np.nan)
        if not (isinstance(price, (int, float)) and np.isfinite(price)):
            price = ti.fast_info.get("last_price", np.nan)
        shares = get_shares_outstanding(ti)
        if pd.notnull(price) and pd.notnull(shares) and shares > 0:
            mcap = float(price) * float(shares)
    except Exception:
        pass
    out["mcap"] = mcap
    # ------------------------------------

    # Quarterly data
    f, c, b = collect_quarterly(t)

    # ------- TTM building blocks with label fallbacks -------
    NI = ttm_first(
        f,
        [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income Applicable To Common Shares",
            "Net Income (Common)",
            "Net income",
            "Net Income From Continuing Operations",
            "Net Income From Continuing Ops",
        ],
    )

    REV = ttm_first(
        f,
        [
            "Total Revenue",
            "Revenue",
            "Revenues",
            "Total revenues",
        ],
    )

    COGS = ttm_first(
        f,
        [
            "Cost Of Revenue",
            "Cost of Revenue",
            "Cost of goods sold",
            "Cost Of Goods And Services Sold",
        ],
    )
    # --------------------------------------------------------

    CFO  = ttm_sum(c, "Total Cash From Operating Activities")
    CAPX = abs(ttm_sum(c, "Capital Expenditures"))
    DIVS = abs(ttm_sum(c, "Cash Dividends Paid"))
    BUYB = abs(ttm_sum(c, "Repurchase Of Stock"))

    # balance sheet averages
    A = q_last(b, "Total Assets", 4)
    E = q_last(b, "Total Stockholder Equity", 4)
    A_avg = float(A.mean()) if isinstance(A, pd.Series) and len(A) > 0 else np.nan
    E_avg = float(E.mean()) if isinstance(E, pd.Series) and len(E) > 0 else np.nan

    # core ratios (valuation + quality)
    out["EP"]      = NI / mcap if (pd.notnull(NI) and pd.notnull(mcap) and mcap > 0) else np.nan
    out["FCP"]     = (CFO - CAPX) / mcap if (pd.notnull(CFO) and pd.notnull(CAPX) and pd.notnull(mcap) and mcap > 0) else np.nan
    out["ShYield"] = (DIVS + BUYB) / mcap  if (pd.notnull(DIVS) and pd.notnull(BUYB) and pd.notnull(mcap) and mcap > 0) else np.nan
    out["ROA"]     = NI / A_avg            if (pd.notnull(NI) and pd.notnull(A_avg) and A_avg != 0) else np.nan
    out["ROE"]     = NI / E_avg            if (pd.notnull(NI) and pd.notnull(E_avg) and E_avg != 0) else np.nan
    out["GPA"]     = ((REV - COGS) / A_avg) if (pd.notnull(REV) and pd.notnull(COGS) and pd.notnull(A_avg) and A_avg != 0) else np.nan

    # asset growth (YoY; 5 last points to compute 4-quarter delta)
    A5 = q_last(b, "Total Assets", 5)
    if isinstance(A5, pd.Series) and len(A5) >= 5 and pd.notnull(A5.iloc[-5]) and A5.iloc[-5] != 0:
        out["AssetGrowth"] = float((A5.iloc[-1] - A5.iloc[-5]) / A5.iloc[-5])
    else:
        out["AssetGrowth"] = np.nan

    # accruals (Sloan light)
    CA   = q_last(b, "Total Current Assets", 5)
    CL   = q_last(b, "Total Current Liabilities", 5)
    CASH = q_last(b, "Cash And Cash Equivalents", 5)
    STD  = q_last(b, "Short Long Term Debt", 5)
    DEP  = q_last(c, "Depreciation", 4)
    try:
        dCA   = CA.iloc[-1]   - CA.iloc[-5]
        dCL   = CL.iloc[-1]   - CL.iloc[-5]
        dCash = CASH.iloc[-1] - CASH.iloc[-5]
        dSTD  = STD.iloc[-1]  - STD.iloc[-5]
        dep4  = float(DEP.sum()) if isinstance(DEP, pd.Series) else 0.0
        accr = ((dCA - dCash) - (dCL - dSTD) - dep4) / A_avg if (pd.notnull(A_avg) and A_avg != 0) else np.nan
        out["Accruals_est"] = float(accr)
    except Exception:
        out["Accruals_est"] = np.nan

    # cache
    try:
        pd.DataFrame([out]).to_parquet(fpath, index=False)
    except Exception:
        pass

    return out

# ----------------------------
# 5) Feature extraction loop (cached, rate-limited)
# ----------------------------
rows = []
for t in tickers:
    try:
        rows.append(features_for(t))
        time.sleep(0.4)  # be gentle for fundamentals
    except Exception as e:
        print("[WARN] fundamentals fail", t, e)

feat = pd.DataFrame(rows).set_index("ticker")
feat.index = feat.index.map(canon)
feat["Accruals_dolt"] = acc_map.reindex(feat.index)
feat["Accruals"] = feat["Accruals_dolt"].where(feat["Accruals_dolt"].notna(), feat["Accruals_est"])

covered = feat["Accruals_dolt"].notna().mean()
print(f"[INFO] Accruals (Dolthub) coverage: {covered:.1%} of universe")
feat[["Accruals_dolt", "Accruals_est", "Accruals"]].to_csv(BASE_DIR / "accruals_compare.csv")
print(f"[INFO] Fundamentals rows: {feat.shape}")

# --- Fill EP via Dolt TTM EPS (fallback) ---
EPS_CSV = BASE_DIR / "ttm_eps.csv"

def _read_eps_csv(path: Path) -> pd.DataFrame:
    # PowerShell ">" writes UTF-16 LE; try a couple encodings
    for enc in ("utf-8", "utf-16", "utf-16le", "utf-16be"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except Exception:
            pass
    # last try: no encoding hint (lets pandas guess)
    return pd.read_csv(path)

if EPS_CSV.exists() and not mpx.empty:
    try:
        eps = _read_eps_csv(EPS_CSV)
        # normalize headers
        eps.columns = [c.strip().lower() for c in eps.columns]
        # expect columns: symbol, ttm_eps
        if "symbol" not in eps.columns or "ttm_eps" not in eps.columns:
            raise ValueError(f"Unexpected EPS CSV columns: {eps.columns.tolist()}")
        eps["symbol"] = eps["symbol"].map(canon)
        eps = eps.dropna(subset=["symbol", "ttm_eps"])
        ttm_eps = eps.set_index("symbol")["ttm_eps"].astype(float)

        # last month-end prices from our price panel
        last_px = mpx.iloc[-1].astype(float)

        # EP_per_share = TTM_EPS / Price
        ep_from_eps = (ttm_eps.reindex(last_px.index) / last_px).replace([np.inf, -np.inf], np.nan)

        # fill only where EP is NA
        if "EP" not in feat.columns:
            feat["EP"] = np.nan
        before = int(feat["EP"].notna().sum())
        feat["EP"] = feat["EP"].where(feat["EP"].notna(), ep_from_eps.reindex(feat.index))
        after = int(feat["EP"].notna().sum())
        print(f"[INFO] EP filled from Dolt EPS for {after - before} tickers (now {after} non-null).")
    except Exception as e:
        print("[WARN] Dolt TTM EPS fallback failed:", e)

# ----------------------------
# 6) Return & risk windows (1y/2y/5y)
# ----------------------------
def max_drawdown_from_returns(r):
    if r.empty:
        return np.nan
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd = (cum / roll_max) - 1.0
    return float(dd.min()) if len(dd) else np.nan

def window_stats(r, months):
    sub = r.dropna().iloc[-months:]
    if sub.empty:
        return (np.nan, np.nan, np.nan)
    tot = float((sub + 1.0).prod() - 1.0)
    vol = float(sub.std() * np.sqrt(12))
    dd  = max_drawdown_from_returns(sub)
    return tot, vol, dd

stats = {}
for t in mpx.columns:
    try:
        r = mret[t].dropna()
        s1 = window_stats(r, 12)
        s2 = window_stats(r, 24)
        s5 = window_stats(r, 60)
        stats[t] = {
            "ret_1y": s1[0], "vol_1y": s1[1], "dd_1y": s1[2],
            "ret_2y": s2[0], "vol_2y": s2[1], "dd_2y": s2[2],
            "ret_5y": s5[0], "vol_5y": s5[1], "dd_5y": s5[2],
        }
    except Exception:
        pass
stats = pd.DataFrame(stats).T

def downside_dev(r):
    r = r.dropna()
    if r.empty:
        return np.nan
    neg = r[r < 0]
    if neg.empty:
        return 0.0
    return float(np.sqrt(np.mean(neg**2)) * np.sqrt(12))

def beta_to_spy(r):
    df = pd.DataFrame({"y": r, "x": spy_ret}).dropna()
    if len(df) < 12:
        return np.nan
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit()
    return float(model.params.get("x", np.nan))

risk = {}
for t in mret.columns:
    try:
        r = mret[t].dropna()
        risk[t] = {
            "downside_dev": downside_dev(r),
            "beta_spy": beta_to_spy(r),
            "max_dd_monthly": max_drawdown_from_returns(r),
        }
    except Exception:
        risk[t] = {"downside_dev": np.nan, "beta_spy": np.nan, "max_dd_monthly": np.nan}
risk = pd.DataFrame(risk).T

# ----------------------------
# 7) Merge, winsorize extrema, z-score, composite
# ----------------------------
X = feat.join(stats, how="inner").join(risk, how="left")

for col in ["EP", "FCP", "ShYield", "ROA", "ROE", "GPA", "AssetGrowth", "Accruals"]:
    if col in X.columns:
        lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = X[col].clip(lower=lo, upper=hi)

Z = X.copy()
for col in ["EP", "FCP", "ShYield", "GPA", "AssetGrowth", "Accruals"]:
    if col in Z.columns:
        Z[col + "_z"] = zscore(Z[col], nan_policy="omit")

Z["Score"] = (
    Z.get("EP_z",          pd.Series(index=Z.index)).fillna(0)
  + Z.get("FCP_z",         pd.Series(index=Z.index)).fillna(0)
  + Z.get("GPA_z",         pd.Series(index=Z.index)).fillna(0)
  - Z.get("AssetGrowth_z", pd.Series(index=Z.index)).fillna(0)
  - Z.get("Accruals_z",    pd.Series(index=Z.index)).fillna(0)
  + Z.get("ShYield_z",     pd.Series(index=Z.index)).fillna(0)
)

# ----------------------------
# 8) Save outputs
# ----------------------------
DATA.mkdir(exist_ok=True, parents=True)
Z.to_csv(DATA / "fundamental_screen.csv")
top = Z.sort_values("Score", ascending=False)
top.head(50).to_csv(DATA / "top_candidates.csv")

print(sp500.head(3)[[sp_col]])
print(nas100.head(3)[[ndx_col]])
print("[DONE] Saved:")
print(f" - {DATA / 'fundamental_screen.csv'}")
print(f" - {DATA / 'top_candidates.csv'}")
