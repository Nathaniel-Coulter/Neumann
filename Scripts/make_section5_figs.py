# make_section5_figs.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold")
DATA = ROOT / r"project\data"
OUT  = ROOT / r"project\figures"
OUT.mkdir(parents=True, exist_ok=True)

FUND_CSV = DATA / "fundamental_screen2.csv"
TOP_CSV  = DATA / "top_candidates2.csv"  # optional

# --- load (tickers are index) ---
df = pd.read_csv(FUND_CSV, index_col=0)

rename_map = {
    "EP":"EP","E_P":"EP","ep":"EP",
    "FCP":"FCP","fcf_yield":"FCP",
    "GPA":"GPA","gross_profit_assets":"GPA",
    "ShYield":"ShYield","shareholder_yield":"ShYield",
    "AssetGrowth":"AssetGrowth","asset_growth":"AssetGrowth",
    "Accruals":"Accruals","accruals":"Accruals",
    "EP_z":"EP_z","FCP_z":"FCP_z","GPA_z":"GPA_z",
    "ShYield_z":"ShYield_z","AssetGrowth_z":"AssetGrowth_z","Accruals_z":"Accruals_z",
    "Score":"Score","CompositeScore":"Score"
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# coerce numerics
for c in ["EP","FCP","GPA","ShYield","AssetGrowth","Accruals",
          "EP_z","FCP_z","GPA_z","ShYield_z","AssetGrowth_z","Accruals_z","Score"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# reconstruct Score if needed
need_z = ["EP_z","FCP_z","GPA_z","ShYield_z","AssetGrowth_z","Accruals_z"]
if "Score" not in df.columns:
    if all(c in df.columns for c in need_z):
        df["Score"] = df["EP_z"] + df["FCP_z"] + df["GPA_z"] - df["AssetGrowth_z"] - df["Accruals_z"] + df["ShYield_z"]
    else:
        raise ValueError("Score missing and z-columns incomplete; cannot reconstruct.")

# ---------- 1) Value–Quality plane ----------
if all(c in df.columns for c in ["EP","GPA","Score"]):
    plt.figure(figsize=(7.6, 5.6))
    sc = plt.scatter(df["EP"], df["GPA"], c=df["Score"], s=22, alpha=0.85)
    cbar = plt.colorbar(sc); cbar.set_label("Composite Score")
    plt.xlabel("Earnings / Price (EP)")
    plt.ylabel("Gross Profit / Assets (GPA)")
    plt.title("Value–Quality Plane (colored by Composite Score)")
    plt.grid(True, linewidth=0.4, alpha=0.5)

    # optional: annotate top candidates if file exists
    if TOP_CSV.exists():
        top = pd.read_csv(TOP_CSV, index_col=0)
        top = top.rename(columns=rename_map)
        for tkr, row in top.head(12).iterrows():
            ep = df.at[tkr, "EP"] if tkr in df.index else np.nan
            gp = df.at[tkr, "GPA"] if tkr in df.index else np.nan
            if pd.notna(ep) and pd.notna(gp):
                plt.annotate(str(tkr), (ep, gp), fontsize=7, xytext=(3,3), textcoords="offset points")

    plt.tight_layout(); plt.savefig(OUT / "value_quality_plane2.png", dpi=220); plt.close()
else:
    print("[WARN] Skipping value_quality_plane: need EP, GPA, Score.")

# ---------- 2) Score distribution ----------
scores = df["Score"].dropna().values
plt.figure(figsize=(7.5, 5.0))
plt.hist(scores, bins=40, alpha=0.85)
q10, q50, q90 = np.quantile(scores, [0.10, 0.50, 0.90])
for x, lab, ls in [(q10,"P10","--"),(q50,"Median","-."),(q90,"P90","--")]:
    plt.axvline(x, linestyle=ls); plt.text(x, plt.ylim()[1]*0.95, lab, ha="center", va="top", rotation=90)
plt.xlabel("Composite Score"); plt.ylabel("Count")
plt.title("Composite Score Distribution (with P10 / Median / P90)")
plt.tight_layout(); plt.savefig(OUT / "score_distribution2.png", dpi=220); plt.close()

# ---------- 3) Top-decile factor composition ----------
if all(c in df.columns for c in need_z):
    top_cut = np.quantile(df["Score"].dropna(), 0.90)
    top_dec = df.loc[df["Score"] >= top_cut, need_z].mean().sort_values(ascending=True)
    plt.figure(figsize=(7.2, 4.6))
    y = np.arange(len(top_dec))
    plt.barh(y, top_dec.values)
    plt.yticks(y, top_dec.index)
    plt.xlabel("Average z-score (Top Decile by Composite Score)")
    plt.title("Top-Decile Factor Composition")
    plt.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    plt.tight_layout(); plt.savefig(OUT / "top_decile_factors2.png", dpi=220); plt.close()
else:
    print("[WARN] Skipping top_decile_factors: missing one or more z-score cols.")

print("[DONE] Saved figures to:", OUT)
