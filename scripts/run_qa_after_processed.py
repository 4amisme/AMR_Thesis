from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DIR, ARTIFACTS_DIR

def qa_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column QA summary."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "na_count": df.isna().sum(),
        "na_percent": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True),
    }).sort_values("na_count", ascending=False)

def main():
    print("QA for processed datasets\n")

    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not files:
        print("No processed CSV files found")
        return

    for f in files:
        print("=" * 80)
        print(f"File: {f.name}")

        df = pd.read_csv(f, low_memory=False)

        print(f"Rows: {df.shape[0]:,}")
        print(f"Columns: {df.shape[1]:,}")

        # Summary NA
        total_na = df.isna().sum().sum()
        print(f"Total NA cells: {total_na:,}")

        # Per-column report
        report = qa_report(df)

        # แสดงเฉพาะคอลัมน์ที่มี NA
        na_cols = report[report["na_count"] > 0]
        if not na_cols.empty:
            print("\n⚠️ Columns with missing values (top 10):")
            print(na_cols.head(10))
        else:
            print("\n✅ No missing values found")

        # save report
        out_dir = ARTIFACTS_DIR / "After_Processing_QA"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{f.stem}_qa.csv"
        report.to_csv(out_path)
        print(f"\nQA report saved to: {out_path}")

    print("\n✅ QA completed for all processed files")


if __name__ == "__main__":
    main()
