import pandas as pd
f = pd.read_csv("project/data/fundamental_screen.csv")

print(f[["EP", "GPA"]].describe())
print(f[["EP", "GPA"]].notna().sum())
