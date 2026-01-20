import pandas as pd
from pathlib import Path

path = Path("data/processed/AllYears_processed.csv")
df = pd.read_csv(path, low_memory=False)

df.shape, df.head(3)

df.columns.tolist()

na_count = df.isna().sum()
na_pct = (df.isna().mean() * 100).round(2)

na_summary = (
    pd.DataFrame({"na_count": na_count, "na_percent": na_pct})
    .sort_values("na_count", ascending=False)
)

na_summary.head(30)   # ดู top 30 ที่ missing เยอะสุด

cols = ["X_HOS_CODE", "X_HOS_NAME", "LABORATORY", "REGION", "WARD_TYPE", "ORGANISM", "ORGANISM_FULL", "AGE_GROUP"]

for c in cols:
    if c in df.columns:
        print("\n===", c, "===")
        print(df[c].value_counts(dropna=False).head(10))
