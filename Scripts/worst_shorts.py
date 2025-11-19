#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Find worst (short) candidates from rolling ML predictions.

Inputs:
  - predictions_rolling CSV (all models ... KFOLD5)  -> per-ticker monthly predictions & outcomes
  - (optional) oos_summary CSV for context           -> not needed for ranking, only printed context
  - (optional) fundamental_screen2 CSV               -> beta, 5y return, drawdown, etc.

Ranking outputs (bottom N):
  1) mean predicted P(up) ↑bad → pick lowest
  2) realized win-rate (fraction of >0) ↑good → pick lowest
  3) consensus rank = average of ranks from (1) & (2), tie-broken by worst avg forward return if available

CLI examples:
  python worst_shorts.py \
    --preds "project/ml_tests/predictions_rolling (all models_shap_importance_calib_KFOLD5).csv" \
    --oos   "project/ml_tests/oos_summary (all models_shap_importance_calib_KFOLD5).csv" \
    --fund  "project/data/fundamental_screen2.csv" \
    --model logistic --horizon 12 --n 20 \
    --out "project/data/short_candidates.csv"

  # If you want to average across all models/horizons:
  python worst_shorts.py --preds "...all models....csv" --n 20 --out short_candidates.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------- column helpers --------------------------

P_UP_CANDIDATES = ["p_up","proba_up","pred_prob","prob_up","p_hat","p"]
LABEL_BOOL_CANDIDATES = ["y","label","target","is_up","up"]
RET_FWD_CANDIDATES = ["ret_fwd","ret_forward","ret_next","ret_next_12m","forward_return"]

MODEL_COLS = ["model","Model","algo","classifier"]
HORIZON_COLS = ["horizon","H","h","label_horizon","horizon_months","Horizon"]

TICKER_COLS = ["ticker","Ticker","symbol","Symbol"]
DATE_COLS   = ["date","Date","asof","as_of","timestamp","ts"]

def pick_first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------- core logic --------------------------

def load_predictions(path_preds, model=None, horizon=None):
    df = pd.read_csv(path_preds)
    # Basic identity
    tcol = pick_first_col(df, TICKER_COLS) or "ticker"
    if tcol not in df.columns:
        raise ValueError("Ticker column not found—please add a 'ticker' column.")

    pcol = pick_first_col(df, P_UP_CANDIDATES)
    if pcol is None:
        raise ValueError("Predicted probability column (P(up)) not found. "
                         f"Looked for: {P_UP_CANDIDATES}")

    # Optional realized label / forward return
    lcol = pick_first_col(df, LABEL_BOOL_CANDIDATES)   # boolean/0-1
    rcol = pick_first_col(df, RET_FWD_CANDIDATES)      # numeric forward return
    dcol = pick_first_col(df, DATE_COLS)

    # Optional filters
    mcol = pick_first_col(df, MODEL_COLS)
    hcol = pick_first_col(df, HORIZON_COLS)

    if model and mcol:
        df = df[df[mcol].astype(str).str.lower() == str(model).lower()]
    if horizon is not None and hcol in df.columns:
        # allow numeric or string
        try:
            df = df[df[hcol].astype(float) == float(horizon)]
        except Exception:
            df = df[df[hcol].astype(str) == str(horizon)]

    # Keep essentials
    keep = [c for c in [tcol, pcol, lcol, rcol, mcol, hcol, dcol] if c and c in df.columns]
    df = df[keep].copy()

    # Normalize column names we’ll use downstream
    ren = {}
    ren[tcol] = "ticker"
    ren[pcol] = "p_up"
    if lcol: ren[lcol] = "label"
    if rcol: ren[rcol] = "ret_fwd"
    if mcol: ren[mcol] = "model"
    if hcol: ren[hcol] = "horizon"
    if dcol: ren[dcol] = "date"
    df = df.rename(columns=ren)

    # If label missing but ret_fwd exists, derive a sign label (>0 = 1, else 0)
    if "label" not in df.columns and "ret_fwd" in df.columns:
        df["label"] = (df["ret_fwd"] > 0).astype(int)

    # Ensure expected columns exist
    if "ticker" not in df.columns or "p_up" not in df.columns:
        raise ValueError("After renaming, missing 'ticker' or 'p_up'.")

    # Drop rows with NA in critical fields
    df = df.dropna(subset=["ticker","p_up"])
    return df

def summarize_by_ticker(df):
    """
    Returns per-ticker aggregates:
      mean_p_up, median_p_up, win_rate, count, mean_ret_fwd (if present)
    """
    group = df.groupby("ticker", as_index=False)
    agg = {
        "p_up": ["mean","median","count"]
    }
    if "label" in df.columns:
        agg["label"] = ["mean"]  # win-rate

    if "ret_fwd" in df.columns:
        agg["ret_fwd"] = ["mean","median"]

    g = group.agg(agg)
    # flatten columns
    g.columns = ["_".join(c).strip("_") for c in g.columns.to_flat_index()]

    # rename to friendly
    rename = {
        "ticker_": "ticker",
        "p_up_mean":"mean_p_up",
        "p_up_median":"median_p_up",
        "p_up_count":"n_obs",
        "label_mean":"win_rate",
        "ret_fwd_mean":"mean_ret_fwd",
        "ret_fwd_median":"median_ret_fwd",
    }
    for k,v in rename.items():
        if k in g.columns:
            g = g.rename(columns={k:v})
    # keep only present
    keep_cols = [c for c in ["ticker","mean_p_up","median_p_up","win_rate","n_obs","mean_ret_fwd","median_ret_fwd"] if c in g.columns]
    return g[keep_cols]

def attach_fundamentals(g, fund_path):
    if not fund_path:
        return g
    f = pd.read_csv(fund_path)
    # Try to find a ticker column in fundamentals
    tcol = pick_first_col(f, TICKER_COLS) or "ticker"
    if tcol not in f.columns:
        return g
    f = f.rename(columns={tcol:"ticker"})
    # Choose a few commonly helpful columns if present
    likely = ["beta","beta_spy","max_dd","max_drawdown","vol_12m","vol_24m","vol_60m",
              "ret_1y","ret_3y","ret_5y","five_y_return","drawdown","downside_dev","sharpe","sortino"]
    extra = [c for c in likely if c in f.columns]
    f_keep = ["ticker"] + extra
    f = f[f_keep].drop_duplicates("ticker")
    return g.merge(f, on="ticker", how="left")

def make_rank_tables(g, n):
    out = {}

    # 1) Worst by mean predicted P(up)
    if "mean_p_up" in g.columns:
        out["worst_by_mean_p_up"] = g.sort_values("mean_p_up", ascending=True).head(n)

    # 2) Worst by realized win rate
    if "win_rate" in g.columns:
        out["worst_by_win_rate"]  = g.sort_values("win_rate", ascending=True).head(n)

    # 3) Consensus: average of ranks (mean_p_up ascending, win_rate ascending),
    #    tie-break by most negative mean_ret_fwd if available.
    if all(c in g.columns for c in ["mean_p_up","win_rate"]):
        tmp = g.copy()
        tmp["rk_p"]   = tmp["mean_p_up"].rank(method="average", ascending=True)
        tmp["rk_win"] = tmp["win_rate"].rank(method="average", ascending=True)
        tmp["consensus_rank"] = (tmp["rk_p"] + tmp["rk_win"]) / 2.0
        if "mean_ret_fwd" in tmp.columns:
            tmp = tmp.sort_values(["consensus_rank","mean_ret_fwd"], ascending=[True, True])
        else:
            tmp = tmp.sort_values("consensus_rank", ascending=True)
        out["worst_by_consensus"] = tmp.head(n).drop(columns=["rk_p","rk_win"])
    return out

def pretty_print_table(title, df):
    print(f"\n=== {title} ===")
    if df.empty:
        print("(no rows)")
        return
    show_cols = [c for c in ["ticker","mean_p_up","win_rate","n_obs","mean_ret_fwd","beta","max_dd","ret_5y"] if c in df.columns]
    if not show_cols:
        show_cols = df.columns.tolist()
    # basic formatting
    fmts = {
        "mean_p_up": "{:.3f}".format,
        "win_rate":  "{:.2f}".format,
        "mean_ret_fwd": "{:.3%}".format,
    }
    print(df[show_cols].to_string(index=False, formatters=fmts))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to predictions_rolling (all models ...) CSV")
    ap.add_argument("--oos", default=None, help="(Optional) Path to oos_summary (all models ...) CSV")
    ap.add_argument("--fund", default=None, help="(Optional) Path to fundamental_screen2.csv")
    ap.add_argument("--model", default=None, help="Filter to a specific model name (e.g., logistic)")
    ap.add_argument("--horizon", type=float, default=None, help="Filter to a specific horizon (e.g., 12)")
    ap.add_argument("--n", type=int, default=20, help="How many worst candidates to list")
    ap.add_argument("--out", default="short_candidates.csv", help="Output CSV path (consensus list + diagnostics)")
    args = ap.parse_args()

    preds = load_predictions(args.preds, model=args.model, horizon=args.horizon)
    per_ticker = summarize_by_ticker(preds)
    per_ticker = attach_fundamentals(per_ticker, args.fund)

    tables = make_rank_tables(per_ticker, args.n)

    # Print to terminal
    for k, v in tables.items():
        pretty_print_table(k.replace("_"," ").title(), v)

    # Save consensus (if available), otherwise fallback to mean_p_up list
    if "worst_by_consensus" in tables:
        out_df = tables["worst_by_consensus"].copy()
    elif "worst_by_mean_p_up" in tables:
        out_df = tables["worst_by_mean_p_up"].copy()
    else:
        # if only win-rate exists
        out_df = tables.get("worst_by_win_rate", per_ticker.sort_values("ticker")).copy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")

    # Optional: print model/horizon context if oos_summary included
    if args.oos:
        try:
            oos = pd.read_csv(args.oos)
            mcol = pick_first_col(oos, MODEL_COLS)
            hcol = pick_first_col(oos, HORIZON_COLS)
            auc_col = None
            for c in ["AUC","auc","mean_auc","MeanAUC","Mean_AUC"]:
                if c in oos.columns:
                    auc_col = c; break
            if mcol and hcol and auc_col:
                ctx = oos.groupby([mcol,hcol], as_index=False)[auc_col].mean()
                print("\n[Context] Mean AUC by model/horizon (from oos_summary):")
                print(ctx.to_string(index=False))
        except Exception as e:
            print(f"\n[warn] Could not parse oos_summary context: {e}")

if __name__ == "__main__":
    main()
