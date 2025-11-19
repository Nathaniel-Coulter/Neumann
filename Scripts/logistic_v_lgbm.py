# logistic_v_lgbm.py (ordered + metrics + coeffs + optional quadratic)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import roc_auc_score, log_loss
from lightgbm import LGBMClassifier

# === Config ===
LOGISTIC_PATH = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\ml_tests\predictions_rolling (Logistic, horizons 12 24 36 kfold 5 platt imp 3).csv"
ALLMODELS_PATH = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\ml_tests\predictions_rolling (all models_shap_importance_calib_KFOLD5).csv"
FUND_PATH      = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\project\data\fundamental_screen.csv"

FEATURES = ["GPA_z", "Accruals_z"]
HORIZON  = 12
USE_QUADRATIC = False  # set True to include [x1,x2,x1^2,x1*x2,x2^2]

# === 1) Load ===
log_df = pd.read_csv(LOGISTIC_PATH)
all_df = pd.read_csv(ALLMODELS_PATH)
fund   = pd.read_csv(FUND_PATH, index_col=0)

for col in ["ticker", "prob_up", "y_true", "test_month", "horizon_m"]:
    if col not in log_df.columns:
        raise ValueError(f"Missing '{col}' in logistic predictions CSV.")
    if col not in all_df.columns:
        raise ValueError(f"Missing '{col}' in all-models predictions CSV.")

log_h = log_df[log_df["horizon_m"] == HORIZON].copy()
lgb_h = all_df[(all_df["horizon_m"] == HORIZON) & (all_df["model"].str.lower() == "lgbm")].copy()

common_months = sorted(set(log_h["test_month"]).intersection(set(lgb_h["test_month"])))
if not common_months:
    raise ValueError("No common test_month between logistic and lgbm for chosen horizon.")
SNAPSHOT_MONTH = common_months[-1]

log_s = log_h[log_h["test_month"] == SNAPSHOT_MONTH].copy()
lgb_s = lgb_h[lgb_h["test_month"] == SNAPSHOT_MONTH].copy()

common_tix = sorted(set(log_s["ticker"]).intersection(set(lgb_s["ticker"])))
log_s = log_s[log_s["ticker"].isin(common_tix)].reset_index(drop=True)
lgb_s = lgb_s[lgb_s["ticker"].isin(common_tix)].reset_index(drop=True)

# === 2) Attach features ===
missing_feats = [f for f in FEATURES if f not in fund.columns]
if missing_feats:
    raise ValueError(f"Missing features in fundamentals file: {missing_feats}")

fund_subset = fund[FEATURES].copy()
fund_subset.index = fund_subset.index.astype(str).str.upper().str.replace('.', '-', regex=False)

def attach_feats(df):
    return df.merge(fund_subset, left_on="ticker", right_index=True, how="left")

log_s = attach_feats(log_s)
lgb_s = attach_feats(lgb_s)

for f in FEATURES:
    log_s = log_s[log_s[f].notna()]
    lgb_s = lgb_s[lgb_s[f].notna()]
log_s = log_s[log_s["y_true"].isin([0, 1])]
lgb_s = lgb_s[lgb_s["y_true"].isin([0, 1])]

common_tix2 = sorted(set(log_s["ticker"]).intersection(set(lgb_s["ticker"])))
log_s = log_s[log_s["ticker"].isin(common_tix2)].reset_index(drop=True)
lgb_s = lgb_s[lgb_s["ticker"].isin(common_tix2)].reset_index(drop=True)

X = log_s[FEATURES].values.astype(float)
y = log_s["y_true"].values.astype(int)

# === 3) Scale (+ optional quadratic) and fit ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if USE_QUADRATIC:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_feat = poly.fit_transform(X_scaled)
    feat_names = poly.get_feature_names_out(FEATURES).tolist()
else:
    X_feat = X_scaled
    feat_names = FEATURES[:]  # ["GPA_z", "Accruals_z"]

# Fit with named columns to avoid LightGBM warning
X_df = pd.DataFrame(X_feat, columns=feat_names)

log_model = LogisticRegression(max_iter=2000)
lgb_model = LGBMClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, random_state=42)

log_model.fit(X_df, y)
lgb_model.fit(X_df, y)

# === 3b) Snapshot metrics ===
p_log = log_model.predict_proba(X_df)[:, 1]
p_lgb = lgb_model.predict_proba(X_df)[:, 1]

from sklearn.metrics import roc_auc_score, log_loss
auc_log = roc_auc_score(y, p_log)
auc_lgb = roc_auc_score(y, p_lgb)
ll_log  = log_loss(y, p_log, labels=[0, 1])
ll_lgb  = log_loss(y, p_lgb, labels=[0, 1])

print(f"[AUC]      Logistic = {auc_log:.3f} | LightGBM = {auc_lgb:.3f}")
print(f"[LogLoss]  Logistic = {ll_log:.4f} | LightGBM = {ll_lgb:.4f}")

# === 3c) Logistic coefficients (scaled + original units) ===
if not USE_QUADRATIC:
    beta_scaled = log_model.coef_.ravel()
    b0_scaled   = log_model.intercept_[0]

    mu = scaler.mean_
    sd = np.sqrt(scaler.var_)
    beta_orig = beta_scaled / sd
    b0_orig   = b0_scaled - np.sum(beta_scaled * (mu / sd))

    # NOTE: use '+.6f' (no space)
    print(f"[β scaled] {FEATURES[0]}={beta_scaled[0]:+.6f}, {FEATURES[1]}={beta_scaled[1]:+.6f}, intercept={b0_scaled:+.6f}")
    print(f"[β orig  ] {FEATURES[0]}={beta_orig[0]:+.6f}, {FEATURES[1]}={beta_orig[1]:+.6f}, intercept={b0_orig:+.6f}")
else:
    names = feat_names
    print("[β (quadratic, scaled)]:")
    for n, c in zip(names, log_model.coef_.ravel()):
        print(f"  {n:>18s}: {c:+.6f}")
    print(f"  {'intercept':>18s}: {log_model.intercept_[0]:+.6f}")

# === 4) Grid + decision boundaries ===
x_min, x_max = X_scaled[:, 0].min() - 1.0, X_scaled[:, 0].max() + 1.0
y_min, y_max = X_scaled[:, 1].min() - 1.0, X_scaled[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220),
                     np.linspace(y_min, y_max, 220))
grid = np.c_[xx.ravel(), yy.ravel()]

if USE_QUADRATIC:
    grid_feat = poly.transform(grid)
else:
    grid_feat = grid

grid_df = pd.DataFrame(grid_feat, columns=feat_names)

Z_log = log_model.predict_proba(grid_df)[:, 1].reshape(xx.shape)
Z_lgb = lgb_model.predict_proba(grid_df)[:, 1].reshape(xx.shape)

# === 5) Plot (vertical, proper aspect) ===
fig, axes = plt.subplots(
    2, 1, figsize=(9, 9), sharex=True, sharey=True, constrained_layout=True
)
titles = [
    f"Linear Boundary (Logistic) — H={HORIZON}m @ {SNAPSHOT_MONTH}",
    f"Non-Linear Boundary (LightGBM) — H={HORIZON}m @ {SNAPSHOT_MONTH}",
]

# Fix identical axes limits across both panels
for ax in axes:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# Plot both panels
panels = []
for ax, Z, title in zip(axes, [Z_log, Z_lgb], titles):
    cs = ax.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.80)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)  # decision boundary
    ax.scatter(
        X_scaled[:, 0], X_scaled[:, 1],
        c=y, cmap="bwr", edgecolor="k", alpha=0.75, s=18
    )
    ax.set_xlabel(f"{FEATURES[0]} (scaled)")
    ax.set_ylabel(f"{FEATURES[1]} (scaled)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    panels.append(cs)

# One colorbar on the right for both axes
fig.colorbar(
    panels[-1], ax=axes.ravel().tolist(), location="right",
    fraction=0.035, pad=0.02, label=r"$\hat{P}(\mathrm{up}\mid x)$"
)

# Save or show
plt.savefig("figures/logistic_v_lgbm_fig2_vertical.png", dpi=300, bbox_inches="tight")
plt.show()
