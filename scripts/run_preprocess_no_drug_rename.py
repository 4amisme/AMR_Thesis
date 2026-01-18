from __future__ import annotations
import pandas as pd

from src.config import (
    RAW_DIR, PROCESSED_DIR,
    YEARS, LISTING_CSV_PATTERN,
    HOSPITAL_MAP_XLSX, DRUG_MAP_XLSX, ORG_MAP_XLSX,
    COL_LAB, COL_WARD_TYPE, COL_AGE, COL_ORGANISM,
)
from src.data.extract import read_listing_csv
from src.data.mappings import load_mappings
from src.data.clean import clean_basic_fields_only
from src.data.dataset_for_no_drug import write_year_output, combine_years


def main():
    print("🚿 Preprocess (NO drug rename, CSV only) started")

    mappings = load_mappings(
        hospital_xlsx=HOSPITAL_MAP_XLSX,
        drug_xlsx=DRUG_MAP_XLSX, 
        org_xlsx=ORG_MAP_XLSX,
    )

    for year in YEARS:
        in_path = RAW_DIR / LISTING_CSV_PATTERN.format(year=year)
        if not in_path.exists():
            print(f"❌ Missing input: {in_path}")
            continue

        print(f"\n=== Preprocess year {year} (no drug rename, CSV) ===")

        df = read_listing_csv(in_path)

        df = clean_basic_fields_only(
            df,
            hos_name_by_lab_code=mappings.hos_name_by_lab_code,
            organism_clean_by_org=mappings.organism_clean_by_org,
            col_laboratory=COL_LAB,
            col_ward_type=COL_WARD_TYPE,
            col_age=COL_AGE,
            col_organism=COL_ORGANISM,
        )

        # ---------- เติม X_HOS_CODE จาก X_HOS_NAME ----------
        if "X_HOS_CODE" in df.columns and "X_HOS_NAME" in df.columns:
            mask = df["X_HOS_CODE"].isna() & df["X_HOS_NAME"].notna()
            if mask.any():
                hos_key = (
                    df["X_HOS_NAME"]
                    .astype(str)
                    .str.strip()
                    .str.replace("โรงพยาบาล", "", regex=False)
                    .str.replace(r"\s+", " ", regex=True)
                )
                df.loc[mask, "X_HOS_CODE"] = hos_key.loc[mask].map(
                    mappings.hos_code_by_hos_name
                )

        out_path = write_year_output(
            df,
            PROCESSED_DIR,
            year,
            fmt="csv"
        )
        print(f"✅ Wrote: {out_path}")

    # ---------- รวมทุกปี (CSV) ----------
    try:
        combined = combine_years(
            PROCESSED_DIR,
            list(YEARS),
            fmt="csv"
        )
        print(f"\n✅ Combined dataset (CSV): {combined}")

    except FileNotFoundError as e:
        print(f"⚠️ Combine skipped: {e}")


if __name__ == "__main__":
    main()