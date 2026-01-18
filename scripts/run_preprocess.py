from __future__ import annotations
import pandas as pd

from src.config import (
    RAW_DIR, PROCESSED_DIR,
    YEARS, LISTING_CSV_PATTERN,
    HOSPITAL_MAP_XLSX, DRUG_MAP_XLSX, ORG_MAP_XLSX,
    COL_LAB,
)

from src.data.extract import read_listing_csv, drop_all_null_columns
from src.data.mappings import load_mappings
from src.data.clean import clean_listing_df
from src.data.dataset import write_year_output, combine_years

def main():
    # Load mapping references once

    mappings = load_mappings(
        hospital_xlsx=HOSPITAL_MAP_XLSX,  # <- Hos_Code-15-Jan.csv
        drug_xlsx=DRUG_MAP_XLSX,          # <- Drug_dict.csv
        org_xlsx=ORG_MAP_XLSX,            # <- mapped_org.csv
    )

    for year in YEARS:
        in_path = RAW_DIR / LISTING_CSV_PATTERN.format(year=year)
        if not in_path.exists():
            print(f"❌ Missing input: {in_path}")
            continue

        print(f"\n=== Preprocess year {year} ===")
        df = read_listing_csv(in_path)
        df = drop_all_null_columns(df)

        # ---------- Core cleaning ----------
        df = clean_listing_df(
            df,
            drug_full_by_whonet5 = mappings.drug_full_by_whonet5,
            organism_clean_by_org = mappings.organism_clean_by_org,
        )

        # ---------- Hospital mapping เติม NaN ----------
        # เติม X_HOS_NAME จาก LABORATORY (ถ้ายังว่าง)
        if "X_HOS_NAME" in df.columns and COL_LAB in df.columns:
            m1 = df["X_HOS_NAME"].isna() & df[COL_LAB].notna()
            if m1.any():
                df.loc[m1, "X_HOS_NAME"] = df.loc[m1, COL_LAB].map(mappings.hos_name_by_lab_code)

        # เติม X_HOS_CODE จาก X_HOS_NAME
        if "X_HOS_CODE" in df.columns and "X_HOS_NAME" in df.columns:
            m2 = df["X_HOS_CODE"].isna() & df["X_HOS_NAME"].notna()
            if m2.any():
                hos_key = (
                    df["X_HOS_NAME"]
                    .astype(str)
                    .str.strip()
                    .str.replace("โรงพยาบาล", "", regex=False)
                    .str.replace(r"\s+", " ", regex=True)
                )
                df.loc[m2, "X_HOS_CODE"] = hos_key.loc[m2].map(mappings.hos_code_by_hos_name)

        # เติม X_HOS_NAME ย้อนกลับจาก X_HOS_CODE ถ้าชื่อว่างแต่มี code
        if "X_HOS_NAME" in df.columns and "X_HOS_CODE" in df.columns:
            m3 = df["X_HOS_NAME"].isna() & df["X_HOS_CODE"].notna()
            if m3.any():
                df.loc[m3, "X_HOS_NAME"] = df.loc[m3, "X_HOS_CODE"].map(mappings.hos_name_by_hos_code)

        out_path = write_year_output(df, PROCESSED_DIR, year, fmt="csv")
        print(f"✅ Wrote: {out_path}")

    # Combine into a single file for handoff
    try:
        combined = combine_years(PROCESSED_DIR, list(YEARS), fmt="csv")
        print(f"\n✅ Combined dataset: {combined}")
    except FileNotFoundError as e:
        print(f"⚠️ Combine skipped: {e}")

if __name__ == "__main__":
    main()