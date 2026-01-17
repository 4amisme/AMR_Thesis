from __future__ import annotations
from pathlib import Path
import pandas as pd


def build_nan_report(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    สร้างรายงาน NaN ของทุกคอลัมน์:
    - nan_count: จำนวน NaN
    - nan_pct: เปอร์เซ็นต์ NaN
    """
    total_rows = len(df)
    nan_count = df.isna().sum().sort_values(ascending=False)

    report = (
        nan_count
        .rename("nan_count")
        .to_frame()
        .assign(
            year=year,
            total_rows=total_rows,
            nan_pct=lambda x: (x["nan_count"] / total_rows * 100).round(4) if total_rows else 0.0
        )
        .reset_index()
        .rename(columns={"index": "column"})
    )
    return report


def print_nan_report(year: int, df: pd.DataFrame, top_n: int = 25) -> None:
    """
    พิมพ์รายงานคอลัมน์ที่มี NaN เยอะสุด (top_n) เพื่อดูเร็วบนหน้าจอ
    """
    rpt = build_nan_report(df, year)
    rpt_nonzero = rpt[rpt["nan_count"] > 0]

    print(f"\n===== NaN Report : {year} =====")
    print(f"Total rows: {len(df)} | Total columns: {df.shape[1]}")
    if rpt_nonzero.empty:
        print("✅ ไม่มี NaN ในทุกคอลัมน์")
    else:
        print(f"Top {min(top_n, len(rpt_nonzero))} columns with NaN:")
        print(rpt_nonzero.head(top_n).to_string(index=False))
    print("================================\n")


def save_nan_report(year: int, df: pd.DataFrame, out_dir: Path) -> Path:
    """
    เซฟรายงานเป็น CSV: artifacts/quality/nan_report_{year}.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rpt = build_nan_report(df, year)

    out_path = out_dir / f"nan_report_{year}.csv"
    rpt.to_csv(out_path, index=False)
    return out_path
