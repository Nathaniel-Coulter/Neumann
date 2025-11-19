import pandas as pd
oos = pd.read_csv(r"project\data\ml_refine\oos_summary.csv")
print("Rows:", len(oos))
print("Mean AUC:", oos["auc"].mean())
print("Mean Long Hit:", oos["long_hit"].mean())
print("Mean Short Hit:", oos["short_hit"].mean())
