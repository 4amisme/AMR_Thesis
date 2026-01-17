from __future__ import annotations
import pandas as pd

from src.config import (
    RAW_DIR, PROCESSED_DIR,
    YEARS, LISTING_CSV_PATTERN,
    HOSPITAL_MAP_XLSX, DRUG_MAP_XLSX, ORG_MAP_XLSX,
    COL_LAB, COL_WARD_TYPE, COL_AGE, COL_ORGANISM,
)
from src.data.extract import read_listing_csv, drop_all_null_columns
from src.data.mappings import load_mappings
from src.data.clean import clean_listing_df
from src.data.dataset import write_year_output, combine_years


def main():
    # Load mapping references once
    # NOTE: ใน config ของแปมเป็น .csv แม้ตัวแปรจะชื่อ *_XLSX ก็ใช้ได้
    mappings = load_mappings(
        hospital_xlsx=HOSPITAL_MAP_XLSX,  # <- ชี้ไปที่ Hos_Code-15-Jan.csv
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

        # ---------- Core cleaning (เดิมของแปม) ----------
        df = clean_listing_df(
            df,
            hos_name_by_lab_code = mappings.hos_name_by_hos_code,
            drug_full_by_whonet5 = mappings.drug_full_by_whonet5,
            organism_clean_by_org = mappings.organism_clean_by_org,
            col_laboratory=COL_LAB,
            col_ward_type=COL_WARD_TYPE,
            col_age=COL_AGE,
            col_organism=COL_ORGANISM,
        )

        # ---------- ✅ Hospital mapping เติม NaN ตาม QA ----------
        # 1) เติม X_HOS_NAME จาก LABORATORY (ถ้ายังว่าง)
        if "X_HOS_NAME" in df.columns and COL_LAB in df.columns:
            m1 = df["X_HOS_NAME"].isna() & df[COL_LAB].notna()
            if m1.any():
                df.loc[m1, "X_HOS_NAME"] = df.loc[m1, COL_LAB].map(mappings.hos_name_by_lab_code)

        # 2) เติม X_HOS_CODE จาก X_HOS_NAME (แก้ปี 2016–2019 ที่ code หายเยอะ)
        # ใช้คีย์ที่ normalize แบบเดียวกับ mapping.py (ใน mappings.hos_code_by_hos_name)
        if "X_HOS_CODE" in df.columns and "X_HOS_NAME" in df.columns:
            m2 = df["X_HOS_CODE"].isna() & df["X_HOS_NAME"].notna()
            if m2.any():
                # normalize แบบง่าย (ให้สอดคล้องกับ _norm_text ที่ตัด "โรงพยาบาล" และจัดช่องว่าง)
                hos_key = (
                    df["X_HOS_NAME"]
                    .astype(str)
                    .str.strip()
                    .str.replace("โรงพยาบาล", "", regex=False)
                    .str.replace(r"\s+", " ", regex=True)
                )
                df.loc[m2, "X_HOS_CODE"] = hos_key.loc[m2].map(mappings.hos_code_by_hos_name)

        # 3) (optional) เติม X_HOS_NAME ย้อนกลับจาก X_HOS_CODE ถ้าชื่อว่างแต่มี code
        if "X_HOS_NAME" in df.columns and "X_HOS_CODE" in df.columns:
            m3 = df["X_HOS_NAME"].isna() & df["X_HOS_CODE"].notna()
            if m3.any():
                df.loc[m3, "X_HOS_NAME"] = df.loc[m3, "X_HOS_CODE"].map(mappings.hos_name_by_hos_code)

        out_path = write_year_output(df, PROCESSED_DIR, year, fmt="parquet")
        print(f"✅ Wrote: {out_path}")

    # Combine into a single file for handoff (amr_2015-2024 / AllYears)
    try:
        combined = combine_years(PROCESSED_DIR, list(YEARS), fmt="parquet")
        print(f"\n✅ Combined dataset: {combined}")
    except FileNotFoundError as e:
        print(f"⚠️ Combine skipped: {e}")

if __name__ == "__main__":
    main()
