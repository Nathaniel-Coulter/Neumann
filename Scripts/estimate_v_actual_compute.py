
dolt = pd.read_csv("/mnt/data/accruals_recent.csv")
est = pd.read_csv("/mnt/data/accruals_quarterly_utf8.csv")


dolt["symbol"] = dolt["symbol"].str.strip().str.upper()
est["symbol"] = est["symbol"].str.strip().str.upper()


estimates = (
    est.groupby("symbol")["wc_accruals_over_assets"]
    .tail(4)
    .groupby(est["symbol"])
    .mean()
    .reset_index()
)


merged = pd.merge(dolt, estimates, on="symbol", how="inner", suffixes=("_dolt", "_est"))


merged["diff"] = merged["accruals_avg_4q"] - merged["wc_accruals_over_assets"]
summary = merged[["accruals_avg_4q", "wc_accruals_over_assets", "diff"]].describe().loc[["mean", "std", "min", "max"]]
summary.round(4)

print(summary.round(4))