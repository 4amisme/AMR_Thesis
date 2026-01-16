from __future__ import annotations
from pathlib import Path

from src.config import (
    RAW_DIR, PROCESSED_DIR, REF_DIR,
    YEARS, LISTING_CSV_PATTERN,
    HOSPITAL_MAP_XLSX, DRUG_MAP_XLSX, ORG_MAP_XLSX,
    HOSPITAL_SHEET, DRUG_SHEET, ORG_SHEET,
    COL_LAB, COL_WARD_TYPE, COL_AGE, COL_ORGANISM,
)
from src.data.extract import read_listing_csv, drop_all_null_columns
from src.data.mappings import load_mappings
from src.data.clean import clean_listing_df
from src.data.dataset import write_year_output, combine_years

def main():
    # Load mapping references once
    mappings = load_mappings(
        hospital_xlsx=HOSPITAL_MAP_XLSX,
        drug_xlsx=DRUG_MAP_XLSX,
        org_xlsx=ORG_MAP_XLSX,
        hospital_sheet=HOSPITAL_SHEET,
        drug_sheet=DRUG_SHEET,
        org_sheet=ORG_SHEET,
    )

    # Process each year independently (memory-friendly)
    for year in YEARS:
        in_path = RAW_DIR / LISTING_CSV_PATTERN.format(year=year)
        if not in_path.exists():
            print(f"❌ Missing input: {in_path}")
            continue

        print(f"\n=== Preprocess year {year} ===")
        df = read_listing_csv(in_path)
        df = drop_all_null_columns(df)

        df = clean_listing_df(
            df,
            hos_name_by_lab_code=mappings.hos_name_by_lab_code,
            drug_full_by_whonet5=mappings.drug_full_by_whonet5,
            organism_clean_by_org=mappings.organism_clean_by_org,
            col_laboratory=COL_LAB,
            col_ward_type=COL_WARD_TYPE,
            col_age=COL_AGE,
            col_organism=COL_ORGANISM,
        )

        out_path = write_year_output(df, PROCESSED_DIR, year, fmt="parquet")
        print(f"✅ Wrote: {out_path}")

    # Optional: combine into a single file for modeling
    try:
        combined = combine_years(PROCESSED_DIR, list(YEARS), fmt="parquet")
        print(f"\n✅ Combined dataset: {combined}")
    except FileNotFoundError as e:
        print(f"⚠️ Combine skipped: {e}")

if __name__ == "__main__":
    main()