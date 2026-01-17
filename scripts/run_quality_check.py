from pathlib import Path
import pandas as pd

from src.config import RAW_DIR, ARTIFACTS_DIR
from src.data.qa_check import print_nan_report, save_nan_report


def load_year_file(year: int) -> pd.DataFrame:
    path = RAW_DIR / f"{year}_listing.csv"
    if not path.exists():
        raise FileNotFoundError(f"❌ ไม่พบไฟล์: {path}")

    return pd.read_csv(path, low_memory=False, dtype=str)


def main():
    years = range(2015, 2025)
    out_dir = ARTIFACTS_DIR / "Nan_QA"

    print("🔎 เริ่มตรวจสอบ NaN ทุกคอลัมน์รายปี...\n")

    for year in years:
        try:
            df = load_year_file(year)

            # แสดงผลบนหน้าจอ (Top 25)
            print_nan_report(year, df, top_n=25)

            # เซฟรายงานเต็มทุกคอลัมน์
            out_path = save_nan_report(year, df, out_dir=out_dir)
            print(f"✅ Saved: {out_path}")

        except Exception as e:
            print(f"⚠️ Year {year} error: {e}")

    print("\n✅ เสร็จสิ้นการตรวจ QA")
    print(f"ดูผลได้ที่: {out_dir}")


if __name__ == "__main__":
    main()
