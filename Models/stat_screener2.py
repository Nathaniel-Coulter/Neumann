# stat_screener2.py
import time, math, numpy as np, pandas as pd
from pathlib import Path
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import zscore

# ----------------------------
# Resolve paths relative to THIS file (…\project\stat_screener2.py)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# ---- Dolthub accruals (preferred) ----
ACC_CSV = BASE_DIR / "accruals_recent.csv"  # columns: symbol, accruals_avg_4q
acc = pd.read_csv(ACC_CSV)

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
    px = yf.download(" ".join(tks), period="10y", auto_adjust=False,
                     threads=True, progress=False, group_by='ticker')
    if isinstance(px.columns, pd.MultiIndex):
        lvl1 = set(px.columns.get_level_values(1))
        key = 'Adj Close' if 'Adj Close' in lvl1 else 'Close'
        out = px.xs(key, axis=1, level=1)
    else:
        out = px.get('Adj Close', px.get('Close', px))
    out = out.loc[:, out.notna().any()]
    return out

BATCH = 35
px_list = []
for i in range(0, len(tickers), BATCH):
    ch = tickers[i:i+BATCH]
    try:
        px_list.append(get_prices(ch))
        time.sleep(1.5)
    except Exception as e:
        print("[WARN] batch fail", ch[:3], e)

if not px_list:
    raise RuntimeError("No price data pulled. Check internet/yfinance.")

prices = pd.concat(px_list, axis=1).sort_index()
prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

# monthly prices & returns (use month-end "ME")
mpx  = prices.resample("ME").last()
mret = mpx.pct_change()

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

spy     = get_adj_close_single("SPY", period="10y")
spy_mpx = spy.resample("ME").last()
spy_ret = spy_mpx.pct_change().dropna()

LAST_PX = {}
if not mpx.empty:
    LAST_PX = mpx.iloc[-1].dropna().to_dict()

# ----------------------------
# 3) Fundamentals helpers
# ----------------------------
def safe_q(t, attr):
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
    if df is None or field not in df.index:
        return np.nan
    s = df.loc[field].sort_index().tail(4).sum()
    return float(s) if pd.notnull(s) else np.nan

def ttm_first(df, candidates):
    for field in candidates:
        v = ttm_sum(df, field)
        if pd.notnull(v):
            return v
    return np.nan

def q_last(df, field, k=4):
    if df is None or field not in df.index:
        return np.nan
    return df.loc[field].sort_index().tail(k)

def get_shares_outstanding(ti: yf.Ticker):
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

def get_sector_safe(ti: yf.Ticker):
    # sector can be missing; that's OK
    try:
        sec = ti.info.get("sector", None)
        if isinstance(sec, str) and sec.strip():
            return sec.strip()
    except Exception:
        pass
    return None

# ----------------------------
# 4) Feature builder (cached)
# ----------------------------
def features_for(t):
    t = canon(t)
    fpath = RAW / f"{t}.parquet"
    if fpath.exists():
        try:
            d = pd.read_parquet(fpath).iloc[0].to_dict()
            # backfill sector if missing in old cache
            if "sector" not in d or (d.get("sector") is None or (isinstance(d.get("sector"), float) and np.isnan(d.get("sector")))):
                try:
                    sec = get_sector_safe(yf.Ticker(t))
                except Exception:
                    sec = None
                d["sector"] = sec
                try:
                    pd.DataFrame([d]).to_parquet(fpath, index=False)
                except Exception:
                    pass
            return d
        except Exception:
            pass  

    out = {"ticker": t}

    mcap = np.nan
    sector = None
    try:
        ti = yf.Ticker(t)
        price = LAST_PX.get(t, np.nan)
        if not (isinstance(price, (int, float)) and np.isfinite(price)):
            price = ti.fast_info.get("last_price", np.nan)
        shares = get_shares_outstanding(ti)
        if pd.notnull(price) and pd.notnull(shares) and shares > 0:
            mcap = float(price) * float(shares)
        sector = get_sector_safe(ti)
    except Exception:
        pass
    out["mcap"] = mcap
    out["sector"] = sector

    f, c, b = collect_quarterly(t)

    NI = ttm_first(
        f, ["Net Income","Net Income Common Stockholders",
            "Net Income Applicable To Common Shares","Net Income (Common)",
            "Net income","Net Income From Continuing Operations",
            "Net Income From Continuing Ops"]
    )

    REV = ttm_first(f, ["Total Revenue","Revenue","Revenues","Total revenues"])
    COGS = ttm_first(f, ["Cost Of Revenue","Cost of Revenue",
                         "Cost of goods sold","Cost Of Goods And Services Sold"])

    CFO  = ttm_sum(c, "Total Cash From Operating Activities")
    CAPX = abs(ttm_sum(c, "Capital Expenditures"))
    DIVS = abs(ttm_sum(c, "Cash Dividends Paid"))
    BUYB = abs(ttm_sum(c, "Repurchase Of Stock"))

    A = q_last(b, "Total Assets", 4)
    E = q_last(b, "Total Stockholder Equity", 4)
    A_avg = float(A.mean()) if isinstance(A, pd.Series) and len(A) > 0 else np.nan
    E_avg = float(E.mean()) if isinstance(E, pd.Series) and len(E) > 0 else np.nan

    out["EP"]      = NI / mcap if (pd.notnull(NI) and pd.notnull(mcap) and mcap > 0) else np.nan
    out["FCP"]     = (CFO - CAPX) / mcap if (pd.notnull(CFO) and pd.notnull(CAPX) and pd.notnull(mcap) and mcap > 0) else np.nan
    out["ShYield"] = (DIVS + BUYB) / mcap  if (pd.notnull(DIVS) and pd.notnull(BUYB) and pd.notnull(mcap) and mcap > 0) else np.nan
    out["ROA"]     = NI / A_avg            if (pd.notnull(NI) and pd.notnull(A_avg) and A_avg != 0) else np.nan
    out["ROE"]     = NI / E_avg            if (pd.notnull(NI) and pd.notnull(E_avg) and E_avg != 0) else np.nan
    out["GPA"]     = ((REV - COGS) / A_avg) if (pd.notnull(REV) and pd.notnull(COGS) and pd.notnull(A_avg) and A_avg != 0) else np.nan

    A5 = q_last(b, "Total Assets", 5)
    if isinstance(A5, pd.Series) and len(A5) >= 5 and pd.notnull(A5.iloc[-5]) and A5.iloc[-5] != 0:
        out["AssetGrowth"] = float((A5.iloc[-1] - A5.iloc[-5]) / A5.iloc[-5])
    else:
        out["AssetGrowth"] = np.nan

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
        time.sleep(0.4)
    except Exception as e:
        print("[WARN] fundamentals fail", t, e)

feat = pd.DataFrame(rows).set_index("ticker")
if "sector" not in feat.columns:
    feat["sector"] = np.nan

feat.index = feat.index.map(canon)
feat["Accruals_dolt"] = acc_map.reindex(feat.index)
feat["Accruals"] = feat["Accruals_dolt"].where(feat["Accruals_dolt"].notna(), feat["Accruals_est"])

covered = feat["Accruals_dolt"].notna().mean()
print(f"[INFO] Accruals (Dolthub) coverage: {covered:.1%} of universe")
_ac_cols = [c for c in ["sector","Accruals_dolt","Accruals_est","Accruals"] if c in feat.columns]
feat[_ac_cols].to_csv(BASE_DIR / "accruals_compare2.csv")

# --- EP fallback via Dolt TTM EPS (after feat exists) ---
EPS_CSV = BASE_DIR / "ttm_eps.csv"

def _read_eps_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-16", "utf-16le", "utf-16be"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except Exception:
            pass
    return pd.read_csv(path)

if EPS_CSV.exists() and not mpx.empty:
    try:
        eps = _read_eps_csv(EPS_CSV)
        eps.columns = [c.strip().lower() for c in eps.columns]
        if "symbol" not in eps.columns or "ttm_eps" not in eps.columns:
            raise ValueError(f"Unexpected EPS CSV columns: {eps.columns.tolist()}")
        eps["symbol"] = eps["symbol"].map(canon)
        eps = eps.dropna(subset=["symbol", "ttm_eps"])
        ttm_eps = eps.set_index("symbol")["ttm_eps"].astype(float)
        last_px = mpx.iloc[-1].astype(float)
        ep_from_eps = (ttm_eps.reindex(last_px.index) / last_px).replace([np.inf, -np.inf], np.nan)
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
# 7) Merge, robust transforms, z-scores, composite
# ----------------------------
X = feat.join(stats, how="inner").join(risk, how="left")

def winsor(s: pd.Series, lo=0.01, hi=0.99):
    s = s.copy()
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=ql, upper=qh)

# Stronger winsorization for value pillars
for col in ["EP", "FCP"]:
    if col in X.columns:
        X[col] = winsor(X[col], 0.025, 0.975)
for col in ["ShYield", "ROA", "ROE", "GPA", "AssetGrowth", "Accruals"]:
    if col in X.columns:
        X[col] = winsor(X[col], 0.01, 0.99)

# Size-neutralize EP and FCP using residuals on log(mcap)
def size_neutralize(colname):
    if colname not in X.columns or "mcap" not in X.columns:
        return
    s = X[colname]
    m = np.log(X["mcap"].replace({0: np.nan}))
    df = pd.DataFrame({"x": s, "m": m}).dropna()
    if len(df) >= 30:
        df = df.copy()
        df["const"] = 1.0
        res = sm.OLS(df["x"], df[["const","m"]]).fit()
        alpha, beta = res.params["const"], res.params["m"]
        fitted = alpha + beta * m
        X[colname + "_sn"] = s - fitted
    else:
        X[colname + "_sn"] = s - s.mean()

size_neutralize("EP")
size_neutralize("FCP")

# Choose the size-neutralized versions for value sleeve; keep ShYield as-is
X["EP_use"]  = X.get("EP_sn",  X.get("EP"))
X["FCP_use"] = X.get("FCP_sn", X.get("FCP"))

# Sector-neutral z (if enough sector coverage), else universe z
def groupwise_z(series, groups):
    df = pd.DataFrame({"x": series, "g": groups}).copy()
    def _z(s):
        if s.std(skipna=True) in (0, np.nan) or s.dropna().shape[0] < 5:
            return (s - s.mean())  # degenerate; returns zeros where available
        return (s - s.mean()) / s.std(ddof=0)
    return df.groupby("g", dropna=False)["x"].transform(_z)

have_sector = X["sector"].notna().mean() >= 0.60 if "sector" in X.columns else False

def z_col(colname):
    if colname not in X.columns:
        return pd.Series(index=X.index, dtype=float)
    s = X[colname]
    if have_sector:
        return groupwise_z(s, X["sector"])
    else:
        return zscore(s, nan_policy="omit")

Z = X.copy()

# Compute z's for pillars (using size-neutralized EP/FCP)
Z["EP_z"]       = pd.Series(z_col("EP_use"), index=Z.index)
Z["FCP_z"]      = pd.Series(z_col("FCP_use"), index=Z.index)
Z["ShYield_z"]  = pd.Series(z_col("ShYield"), index=Z.index)
Z["GPA_z"]      = pd.Series(z_col("GPA"), index=Z.index)
Z["Accruals_z"] = pd.Series(z_col("Accruals"), index=Z.index)
Z["AssetGrowth_z"] = pd.Series(z_col("AssetGrowth"), index=Z.index)

# Soft cap to prevent domination (smooth, not a hard clip)
def soft_cap(zs):
    return np.tanh(zs / 2.0) * 3.0  # ~±3 max

for c in ["EP_z","FCP_z","ShYield_z","GPA_z","Accruals_z","AssetGrowth_z"]:
    if c in Z.columns:
        Z[c] = soft_cap(Z[c])

# Composite (Value/Quality/Stability)
Z["Value"]    = Z[["EP_z","FCP_z","ShYield_z"]].mean(axis=1, skipna=True)
Z["Quality"]  = Z[["GPA_z"]].mean(axis=1, skipna=True) - Z[["Accruals_z"]].mean(axis=1, skipna=True)
Z["Stability"]= -Z["AssetGrowth_z"]

Z["Score2"]   = 0.40*Z["Value"].fillna(0) + 0.40*Z["Quality"].fillna(0) + 0.20*Z["Stability"].fillna(0)

# ----------------------------
# 8) Save outputs (with "2")
# ----------------------------
DATA.mkdir(exist_ok=True, parents=True)
out_cols = [
    "sector","mcap","EP","FCP","EP_sn","FCP_sn","EP_use","FCP_use","ShYield","GPA",
    "AssetGrowth","Accruals","EP_z","FCP_z","ShYield_z","GPA_z","Accruals_z",
    "AssetGrowth_z","Value","Quality","Stability","Score2",
    "beta_spy","downside_dev","max_dd_monthly","ret_5y","vol_5y","dd_5y"
]
present = [c for c in out_cols if c in Z.columns]
Z[present].to_csv(DATA / "fundamental_screen2.csv")

top2 = Z.sort_values("Score2", ascending=False)
top2.head(50)[present].to_csv(DATA / "top_candidates2.csv")

print(sp500.head(3)[[sp_col]])
print(nas100.head(3)[[ndx_col]])
print("[DONE] Saved:")
print(f" - {DATA / 'fundamental_screen2.csv'}")
print(f" - {DATA / 'top_candidates2.csv'}")
print(f"Sector coverage used: {have_sector}")
