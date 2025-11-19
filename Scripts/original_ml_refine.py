# ml_refine.py
# Rolling OOS classifier with toggles:
# - Models: logistic, mlp (PyTorch), lgbm (LightGBM)
# - K-fold inside each train window for tuning/robustness
# - Calibration: none | platt | isotonic
# - Multiple horizons (12, 24, 36 months)
# - Long/Short evaluation (top-N and bottom-N)
#
# Inputs (already created by your screener run):
#   project/data/fundamental_screen.csv
# Optional cache (auto-created):
#   project/data/monthly_prices.parquet  (to build labels)
#
# Usage examples (PowerShell):
#   python .\project\ml_refine.py --models logistic mlp lgbm --calibration platt --kfold 5
#   python .\project\ml_refine.py --models logistic --calibration isotonic --horizons 12 24
#   python .\project\ml_refine.py --models lgbm --no-download  # if prices cached
#
# Notes:
# - Requires: pandas, numpy, scikit-learn, yfinance, statsmodels, torch (for MLP), lightgbm (for LGBM)
# - If lightgbm isn’t installed, the script will warn and skip that model.
# - CUDA: MLP auto-uses CUDA if available.
# - Outputs are written under project/data/ml_refine/

import os, sys, math, time, argparse, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

# yfinance for price labels
import yfinance as yf

# torch for MLP
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = DATA_DIR / "ml_refine"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FUND_CSV = DATA_DIR / "fundamental_screen.csv"
PRICES_PARQ = DATA_DIR / "monthly_prices.parquet"  # cache

# --- NEW: optional SHAP
try:
    import shap
    _SHAP_OK = True
except Exception:
    _SHAP_OK = False

INTERPRET_DIR = OUT_DIR / "interpret"
INTERPRET_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def canon(t):
    return str(t).strip().upper().replace('.', '-')

def load_fundamentals():
    df = pd.read_csv(FUND_CSV, index_col=0)
    df.index = df.index.map(canon)
    return df

def get_universe_from_fund(df):
    return sorted(list(df.index.unique()))

def download_monthly_prices(tickers, years=10, allow_download=True):
    """
    Returns monthly Adj Close for all tickers; caches to parquet.
    """
    if PRICES_PARQ.exists():
        try:
            px = pd.read_parquet(PRICES_PARQ)
            # validate coverage
            have = set(px.columns)
            miss = [t for t in tickers if t not in have]
            if allow_download and miss:
                add = yf.download(" ".join(miss), period=f"{years}y", auto_adjust=False,
                                  progress=False, group_by="ticker")
                if isinstance(add.columns, pd.MultiIndex):
                    lvl1 = set(add.columns.get_level_values(1))
                    key = "Adj Close" if "Adj Close" in lvl1 else "Close"
                    add = add.xs(key, axis=1, level=1)
                else:
                    add = add.get("Adj Close", add.get("Close", add))
                add = add.loc[:, add.notna().any()]
                px = px.join(add, how="outer")
                px = px.sort_index()
                px = px.resample("ME").last()
                px.to_parquet(PRICES_PARQ)
            return px.resample("ME").last()
        except Exception:
            pass

    if not allow_download:
        raise RuntimeError("Price cache missing and downloads disabled (--no-download).")

    # fresh download in batches
    B = 35
    batches = [tickers[i:i+B] for i in range(0, len(tickers), B)]
    parts = []
    for ch in batches:
        try:
            d = yf.download(" ".join(ch), period=f"{years}y", auto_adjust=False,
                            progress=False, group_by="ticker")
            if isinstance(d.columns, pd.MultiIndex):
                lvl1 = set(d.columns.get_level_values(1))
                key = "Adj Close" if "Adj Close" in lvl1 else "Close"
                d = d.xs(key, axis=1, level=1)
            else:
                d = d.get("Adj Close", d.get("Close", d))
            d = d.loc[:, d.notna().any()]
            parts.append(d)
            time.sleep(1.0)
        except Exception as e:
            print("[WARN] px batch fail", ch[:3], e)

    if not parts:
        raise RuntimeError("No monthly prices pulled.")
    px = pd.concat(parts, axis=1)
    px = px.loc[:, ~px.columns.duplicated(keep="first")]
    px = px.sort_index().resample("ME").last()
    px.to_parquet(PRICES_PARQ)
    return px

def forward_total_return(px_series, months):
    """ (P_{t+M} / P_t) - 1 """
    return (px_series.shift(-months) / px_series) - 1.0

def winsorize_srs(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

def compute_labels(px_m, horizons=(12,)):
    """
    Returns dict: {h: DataFrame of fwd returns per month x ticker}, and a stacked label df.
    """
    fwdrets = {}
    for h in horizons:
        f = px_m.apply(lambda col: forward_total_return(col, h))
        fwdrets[h] = f
    return fwdrets

def make_feature_matrix(fund_df):
    """
    Select and clean features from fundamental_screen.csv
    """
    feats = [
        "EP_z", "FCP_z", "GPA_z", "Accruals_z", "AssetGrowth_z", "ShYield_z",
        "beta_spy", "downside_dev", "max_dd_monthly"
    ]
    X = fund_df.copy()
    # Ensure presence; fill if missing
    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
    # Winsorize primary raw z’s lightly (they’re z-scores already; this is protective)
    for c in ["EP_z", "FCP_z", "GPA_z", "Accruals_z", "AssetGrowth_z", "ShYield_z"]:
        if c in X.columns:
            X[c] = winsorize_srs(X[c].astype(float), 0.01)
    # Risk metrics may be wide; winsorize
    for c in ["beta_spy", "downside_dev", "max_dd_monthly"]:
        if c in X.columns:
            X[c] = winsorize_srs(X[c].astype(float), 0.01)
    X = X[feats]
    X = X.astype(float).fillna(0.0)  # <= add this
    return X

# ---------------------------
# Models
# ---------------------------
def build_logreg(C=0.1, class_weight=None):
    return LogisticRegression(
        C=C, solver="lbfgs", penalty="l2",
        max_iter=2000, n_jobs=None, class_weight=class_weight
    )

class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, d_in, hidden=64, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_val=None, y_val=None, epochs=100, batch=128, lr=1e-3, device="cpu"):
    model = MLP(d_in=X_train.shape[1], hidden=64, p=0.10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TabDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    if X_val is not None:
        ds_val = TabDataset(X_val, y_val)
        dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False)

    best_state, best_val = None, -np.inf
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(X_val.astype(np.float32)).to(device)
                logits = model(xb).cpu().numpy().ravel()
                probs = 1/(1+np.exp(-logits))
                auc = roc_auc_score(y_val, probs)
                if auc > best_val:
                    best_val = auc
                    best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_mlp(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(xb).cpu().numpy().ravel()
        return 1/(1+np.exp(-logits))

def has_lightgbm():
    try:
        import lightgbm as lgb
        return True
    except Exception:
        return False

def train_lgbm(X_train, y_train, X_val=None, y_val=None, seed=42):
    import lightgbm as lgb
    params = dict(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_data_in_leaf=10,
        verbose=-1,
        seed=seed,
    )
    ltrain = lgb.Dataset(X_train, label=y_train)
    valid_sets = [ltrain]
    valid_names = ["train"]
    if X_val is not None:
        lval = lgb.Dataset(X_val, label=y_val)
        valid_sets.append(lval); valid_names.append("val")

    model = lgb.train(
        params,
        ltrain,
        num_boost_round=800,
        valid_sets=valid_sets,
        valid_names=valid_names,
        early_stopping_rounds=75 if X_val is not None else None,
        verbose_eval=False,
    )
    return model

def predict_lgbm(model, X):
    return model.predict(X, num_iteration=model.best_iteration)

# ---------------------------
# Interpretability helpers  (NEW)
# ---------------------------
def _safe_norm_series(s: pd.Series) -> pd.Series:
    s = s.fillna(0.0)
    tot = float(np.abs(s).sum())
    if tot == 0:
        return s
    return (np.abs(s) / tot)

def logreg_importance(est, feature_names):
    """
    Handles plain LogisticRegression or CalibratedClassifierCV wrapping a logistic.
    Returns Series of normalized |coef|.
    """
    base = est
    # If calibrated, try to extract the underlying estimator
    if hasattr(est, "base_estimator"):
        base = est.base_estimator
    elif hasattr(est, "calibrated_classifiers_"):
        # pick the first calibrated classifier's base estimator if available
        try:
            base = est.calibrated_classifiers_[0].base_estimator
        except Exception:
            pass
    if not hasattr(base, "coef_"):
        return pd.Series(0.0, index=feature_names)
    coefs = np.ravel(base.coef_)
    imp = pd.Series(np.abs(coefs), index=feature_names)
    return _safe_norm_series(imp)

def lgbm_importance(model, feature_names):
    try:
        gain = model.feature_importance(importance_type="gain")
        s = pd.Series(gain, index=feature_names)
        return _safe_norm_series(s)
    except Exception:
        return pd.Series(0.0, index=feature_names)

def _torch_prob_wrapper(mlp_model):
    # Wraps the MLP to return probabilities for SHAP
    class _Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            logits = self.inner(x)
            return torch.sigmoid(logits)
    return _Wrap(mlp_model)

def shap_summary(model_kind, model_obj, X_sample, feature_names, device="cpu"):
    """
    Returns Series: mean |SHAP| per feature. Uses a small Kernel/Deep explainer.
    - model_kind: 'logistic' | 'mlp' | 'lgbm'
    """
    if not _SHAP_OK:
        return None, "skipped: shap not installed"

    try:
        if model_kind == "mlp":
            # Torch DeepExplainer on small background
            # X_sample is numpy; convert to torch tensor
            xb = torch.from_numpy(X_sample.astype(np.float32)).to(device)
            wrapped = _torch_prob_wrapper(model_obj).to(device)
            # DeepExplainer expects background; use first 128 rows (or all if smaller)
            bg_n = min(128, xb.shape[0])
            expl = shap.DeepExplainer(wrapped, xb[:bg_n])
            # Explain a slice (same as background or up to 512 points)
            xs_n = min(512, xb.shape[0])
            shap_vals = expl.shap_values(xb[:xs_n])
            # For binary classification with sigmoid output, shap_values is a tensor
            if isinstance(shap_vals, list):
                V = shap_vals[0]
            else:
                V = shap_vals
            V = V.detach().cpu().numpy()
            s = pd.Series(np.abs(V).mean(axis=0), index=feature_names)
            return s, None

        elif model_kind in ("logistic", "lgbm"):
            # Use KernelExplainer (works with predict_proba)
            # Build a callable that outputs prob(class=1)
            def fprob(X):
                X = np.asarray(X, dtype=np.float32)
                if model_kind == "logistic":
                    if hasattr(model_obj, "predict_proba"):
                        return model_obj.predict_proba(X)[:,1]
                    else:
                        logits = model_obj.decision_function(X)
                        return 1/(1+np.exp(-logits))
                else:  # lgbm
                    return model_obj.predict(X)
            # background + sample
            bg_n = min(128, X_sample.shape[0])
            xs_n = min(512, X_sample.shape[0])
            expl = shap.KernelExplainer(fprob, X_sample[:bg_n])
            shap_vals = expl.shap_values(X_sample[:xs_n], nsamples="auto")
            # shap_vals is array [n, d]
            V = np.asarray(shap_vals)
            s = pd.Series(np.abs(V).mean(axis=0), index=feature_names)
            return s, None

        else:
            return None, f"unsupported model_kind {model_kind}"

    except Exception as e:
        return None, f"shap error: {e}"

# ---------------------------
# Rolling OOS engine
# ---------------------------
def build_panel_for_month(fund_df, feat_df, month_end):
    """
    Fund data are cross-sectional (no panel). We assume the most recent snapshot.
    If you later store time-stamped fundamentals, adapt here.
    """
    # For now, just return the same feature row set for all months.
    # Index: tickers
    return feat_df.copy()

def prepare_xy(panel_X, px_m, month, horizon):
    """
    Align X (tickers) with label y at month for given horizon.
    Label = 1{ fwd total return > 0 }
    """
    # price label
    if month not in px_m.index:
        return None, None, None
    # fwd return
    fwd = (px_m.shift(-horizon) / px_m - 1.0).loc[month]
    y = (fwd > 0).astype(float)

    # Align tickers
    common = panel_X.index.intersection(fwd.dropna().index)
    if len(common) < 50:  # too small to be useful
        return None, None, None
    X = panel_X.loc[common]
    y = y.loc[common]
    return X, y, common

def run_kfold_calibrated(base_clf, X, y, kfold, calibration, random_state=42):
    """
    Fits a base classifier, optionally wrapped in a calibration layer.
    Returns a fitted estimator with predict_proba(X) behavior.
    """
    # We do a simple K-fold for hyperparam stability (no grid in this stub).
    # You could extend: try multiple C for logistic, etc., and pick best mean AUC.
    kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
    aucs = []
    for tr, va in kf.split(X):
        Xtr, Xva = X[tr], X[va]; ytr, yva = y[tr], y[va]
        model = clone(base_clf)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:,1] if hasattr(model, "predict_proba") else model.decision_function(Xva)
        if p.ndim == 1 and p.min()>=0 and p.max()<=1:
            preds = p
        else:
            # safety: map to 0..1 via sigmoid if decision_function
            preds = 1/(1+np.exp(-p))
        aucs.append(roc_auc_score(yva, preds))
    # Train final on full
    base_clf.fit(X, y)
    est = base_clf
    if calibration in ("platt","isotonic"):
        method = "sigmoid" if calibration=="platt" else "isotonic"
        est = CalibratedClassifierCV(base_estimator=base_clf, method=method, cv=min(5, kfold))
        est.fit(X, y)
    return est, np.mean(aucs)

def evaluate_longshort(probs, y_true, tickers, top_n=10):
    """
    Returns dict of long/short performance for this month:
    - long basket = top_n by prob
    - short basket = bottom_n by prob
    Metrics use realized sign (y_true as 0/1) & naive return proxy:
      long_hit = mean(y in long), short_hit = mean(1-y in short)
    """
    df = pd.DataFrame({"p": probs, "y": y_true.values}, index=tickers)
    df = df.sort_values("p", ascending=False)
    longs = df.head(top_n)
    shorts = df.tail(top_n)

    # Hit rates
    long_hit = longs["y"].mean()
    short_hit = (1.0 - shorts["y"]).mean()

    return {
        "long_hit": float(long_hit),
        "short_hit": float(short_hit),
        "long_names": ";".join(longs.index.tolist()),
        "short_names": ";".join(shorts.index.tolist()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["logistic"], choices=["logistic","mlp","lgbm","all"])
    ap.add_argument("--calibration", default="none", choices=["none","platt","isotonic"])
    ap.add_argument("--kfold", type=int, default=1, help="K-fold inside each train window (>=1).")
    ap.add_argument("--train_months", type=int, default=60)
    ap.add_argument("--test_step", type=int, default=1, help="slide by N months")
    ap.add_argument("--oos_span", type=int, default=6, help="evaluate AUC on the first month of each test span; predictions saved monthly regardless")
    ap.add_argument("--horizons", nargs="+", type=int, default=[12], help="label horizons in months")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--no-download", action="store_true", help="disallow downloading prices (require cache)")
    ap.add_argument("--explain", action="store_true", help="dump feature importance and SHAP summaries periodically")
    ap.add_argument("--imp-every", type=int, default=3, help="compute interpretability every N test months (default 3)")
    ap.add_argument("--explain-max", type=int, default=512, help="max rows for SHAP sampling (default 512)")
    args = ap.parse_args()

    want_models = ["logistic","mlp","lgbm"] if "all" in args.models else args.models

    fund = load_fundamentals()
    tickers = get_universe_from_fund(fund)
    print(f"[INFO] Universe size: {len(tickers)}")

    # Monthly prices for labels
    px_m = download_monthly_prices(tickers, years=10, allow_download=not args.no_download)
    px_m = px_m.loc[:, px_m.columns.intersection(tickers)]
    fwd = compute_labels(px_m, horizons=tuple(args.horizons))

    # Static feature matrix (cross-sectional snapshot)
    feat_df = make_feature_matrix(fund)

    # Rolling windows bounds (by month index)
    months = px_m.index.sort_values()
    # Start after enough history for forward return max horizon
    max_h = max(args.horizons)
    start_idx = args.train_months
    end_idx = len(months) - max_h - 1
    if end_idx <= start_idx:
        raise RuntimeError("Not enough history for requested horizons/train window.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "mlp" in want_models:
        print(f"[INFO] Torch device: {device}")

    # storage
    all_preds = []  # rows: dict per month/model/horizon with per-ticker probs
    oos_rows  = []  # per test month summary

    # model availability
    lgbm_ok = has_lightgbm()
    if "lgbm" in want_models and not lgbm_ok:
        print("[WARN] lightgbm not installed; skipping lgbm model.")
        want_models = [m for m in want_models if m != "lgbm"]

    # rolling loop
    for i in range(start_idx, end_idx, args.test_step):
        train_start = i - args.train_months
        train_end   = i  # exclusive
        test_m      = months[i]

        # Snapshot panel X for this test month
        panel_X = build_panel_for_month(fund, feat_df, test_m)
        feat_names = list(panel_X.columns)

        for H in args.horizons:
            # Build train set: months [train_start : train_end) for which labels exist
            # We train on cross-sectional data each month in train span; stack them
            X_list, y_list = [], []
            idx_list = []
            for j in range(train_start, train_end):
                m = months[j]
                X, y, common = prepare_xy(panel_X, px_m, m, H)
                if X is None: 
                    continue
                X_list.append(X.values)
                y_list.append(y.values.astype(float))
                idx_list.append((m, common))
            if not X_list:
                continue
            X_train = np.vstack(X_list)
            y_train = np.concatenate(y_list)

            # Prepare test X/y for current month
            X_t, y_t, tick_t = prepare_xy(panel_X, px_m, test_m, H)
            if X_t is None:
                continue

            # Scale (fit on TRAIN only)
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_t.values)

            # Fit per selected model(s)
            model_results = {}
            for mdl in want_models:
                if mdl == "logistic":
                    base = build_logreg(C=0.1)
                    if args.kfold > 1 or args.calibration != "none":
                        est, cv_auc = run_kfold_calibrated(base, X_train_s, y_train, max(2,args.kfold), args.calibration)
                    else:
                        base.fit(X_train_s, y_train)
                        est = base; cv_auc = np.nan
                    # predict proba
                    if hasattr(est, "predict_proba"):
                        p_test = est.predict_proba(X_test_s)[:,1]
                    else:
                        # decision_function -> sigmoid
                        logits = est.decision_function(X_test_s)
                        p_test = 1/(1+np.exp(-logits))
                    model_results["logistic"] = (p_test, cv_auc)

                elif mdl == "mlp":
                    # simple split inside train for early stopping proxy
                    # (if K-fold requested, we still use it only for logreg; here keep it simple)
                    n = X_train_s.shape[0]
                    val_cut = int(0.85*n)
                    Xtr, Xva = X_train_s[:val_cut], X_train_s[val_cut:]
                    ytr, yva = y_train[:val_cut], y_train[val_cut:]
                    mlp = train_mlp(Xtr, ytr, Xva, yva, epochs=200, batch=256, lr=3e-4, device=device)
                    p_test = predict_mlp(mlp, X_test_s, device=device)
                    # Optional calibration on top of MLP:
                    if args.calibration in ("platt","isotonic"):
                        # fit calibrator on train preds (out-of-sample preds are better; here we use val as proxy)
                        # produce train preds
                        p_tr = predict_mlp(mlp, Xtr, device=device)
                        # wrap with a simple logistic link via CalibratedClassifierCV is non-trivial for raw probs,
                        # so we do a quick Platt-scale fit ourselves if 'platt', or isotonic via sklearn.
                        if args.calibration == "platt":
                            # Fit platt: sigmoid on logit(p_tr) vs ytr
                            eps = 1e-6
                            logit_tr = np.log(np.clip(p_tr,eps,1-eps)) - np.log(1-np.clip(p_tr,eps,1-eps))
                            # fit logistic on (logit_tr -> ytr)
                            lr_pl = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                            lr_pl.fit(logit_tr.reshape(-1,1), ytr)
                            # apply on test
                            logit_te = np.log(np.clip(p_test,eps,1-eps)) - np.log(1-np.clip(p_test,eps,1-eps))
                            p_test = lr_pl.predict_proba(logit_te.reshape(-1,1))[:,1]
                        else:
                            # isotonic: use sklearn IsotonicRegression
                            from sklearn.isotonic import IsotonicRegression
                            ir = IsotonicRegression(out_of_bounds="clip")
                            ir.fit(p_tr, ytr)
                            p_test = ir.transform(p_test)
                    model_results["mlp"] = (p_test, np.nan)

                elif mdl == "lgbm":
                    # LightGBM with a small validation split
                    n = X_train_s.shape[0]
                    val_cut = int(0.85*n)
                    Xtr, Xva = X_train_s[:val_cut], X_train_s[val_cut:]
                    ytr, yva = y_train[:val_cut], y_train[val_cut:]
                    lgbm_model = train_lgbm(Xtr, ytr, X_va=Xva, y_val=yva)
                    p_test = predict_lgbm(lgbm_model, X_test_s)
                    # simple calibration option on LGBM
                    if args.calibration in ("platt","isotonic"):
                        # Fit calibrator on validation predictions
                        p_va = predict_lgbm(lgbm_model, Xva)
                        if args.calibration == "platt":
                            eps = 1e-6
                            logit_va = np.log(np.clip(p_va,eps,1-eps)) - np.log(1-np.clip(p_va,eps,1-eps))
                            lr_pl = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                            lr_pl.fit(logit_va.reshape(-1,1), yva)
                            logit_te = np.log(np.clip(p_test,eps,1-eps)) - np.log(1-np.clip(p_test,eps,1-eps))
                            p_test = lr_pl.predict_proba(logit_te.reshape(-1,1))[:,1]
                        else:
                            from sklearn.isotonic import IsotonicRegression
                            ir = IsotonicRegression(out_of_bounds="clip")
                            ir.fit(p_va, yva)
                            p_test = ir.transform(p_test)
                    model_results["lgbm"] = (p_test, np.nan)

            # evaluate, record
            for mdl, (p_test, cv_auc) in model_results.items():
                # AUC this month (classification quality)
                auc = roc_auc_score(y_t.values, p_test) if len(np.unique(y_t.values))>1 else np.nan
                # Long/short eval
                ls = evaluate_longshort(p_test, y_t, tickers=tick_t, top_n=args.topn)

                            # --- NEW: Interpretability dumps (periodic)
                if args.explain and ((i - start_idx) % max(1, args.imp_every) == 0):
                    date_tag = str(test_m.date())
                    # 1) Feature importance
                    if mdl == "logistic":
                        imp_s = logreg_importance(est, feat_names)
                    elif mdl == "lgbm":
                        try:
                            imp_s = lgbm_importance(lgbm_model, feat_names)
                        except Exception:
                            imp_s = pd.Series(0.0, index=feat_names)
                    else:  # mlp: use SHAP only; no native "importance"
                        imp_s = None

                    if imp_s is not None:
                        imp_df = pd.DataFrame({"feature": imp_s.index, "importance_norm": imp_s.values})
                        imp_df = imp_df.sort_values("importance_norm", ascending=False)
                        imp_path = INTERPRET_DIR / f"imp_{mdl}_H{H}_{date_tag}.csv"
                        imp_df.to_csv(imp_path, index=False)

                    # 2) SHAP (mean |SHAP| per feature) on a capped sample of the test design matrix
                    # Use test month (X_test_s) for local attribution around current decision boundary
                    expl_cap = min(args.explain_max, X_test_s.shape[0])
                    X_sample = X_test_s[:expl_cap].copy()

                    model_for_shap = None
                    model_kind = None
                    if mdl == "logistic":
                        model_for_shap = est
                        model_kind = "logistic"
                    elif mdl == "mlp":
                        model_for_shap = mlp
                        model_kind = "mlp"
                    elif mdl == "lgbm":
                        model_for_shap = lgbm_model
                        model_kind = "lgbm"

                    shap_s, shap_err = shap_summary(model_kind, model_for_shap, X_sample, feat_names, device=device)
                    if shap_s is not None:
                        shap_df = pd.DataFrame({"feature": shap_s.index, "mean_abs_shap": shap_s.values})
                        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
                        shap_path = INTERPRET_DIR / f"shap_{mdl}_H{H}_{date_tag}.csv"
                        shap_df.to_csv(shap_path, index=False)
                    elif shap_err:
                        # leave a breadcrumb so you know why a particular month/model didn't produce SHAP
                        with open(INTERPRET_DIR / f"shap_{mdl}_H{H}_{date_tag}.log", "w", encoding="utf-8") as fh:
                            fh.write(shap_err + "\n")

                oos_rows.append({
                    "test_month": str(test_m.date()),
                    "horizon_m": H,
                    "model": mdl,
                    "cv_auc_mean": None if (isinstance(cv_auc,float) and math.isnan(cv_auc)) else cv_auc,
                    "auc": None if (isinstance(auc,float) and math.isnan(auc)) else auc,
                    "long_hit": ls["long_hit"],
                    "short_hit": ls["short_hit"],
                    "topn": args.topn
                })

                # store per-ticker predictions for this month
                pred_df = pd.DataFrame({
                    "ticker": tick_t,
                    "prob_up": p_test,
                    "y_true": y_t.values
                })
                pred_df["test_month"] = str(test_m.date())
                pred_df["horizon_m"] = H
                pred_df["model"] = mdl
                all_preds.append(pred_df)

        # optional: progress
        # print(f"[INFO] done {test_m.date()}")

    if all_preds:
        preds = pd.concat(all_preds, ignore_index=True)
        preds_path = OUT_DIR / "predictions_rolling.csv"
        preds.to_csv(preds_path, index=False)
        print(f"[DONE] Saved predictions -> {preds_path}")

    if oos_rows:
        oos = pd.DataFrame(oos_rows)
        oos_path = OUT_DIR / "oos_summary.csv"
        oos.to_csv(oos_path, index=False)
        print(f"[DONE] Saved OOS summary -> {oos_path}")

if __name__ == "__main__":
    main()