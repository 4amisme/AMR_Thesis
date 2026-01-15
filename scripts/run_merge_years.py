from __future__ import annotations
from src.config import YEARS, PROCESSED_DIR
from src.data.dataset import combine_years

def main():
    out = combine_years(PROCESSED_DIR, list(YEARS), fmt="parquet")
    print(f"✅ Wrote combined dataset: {out}")

if __name__ == "__main__":
    main()
