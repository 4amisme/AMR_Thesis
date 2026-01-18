from __future__ import annotations
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_sqlite"                # WHONET-exported listing CSVs (per year)
PROCESSED_DIR = DATA_DIR / "processed"    # cleaned outputs (per year + combined)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REF_DIR = PROJECT_ROOT / "references"     # mapping reference files (tracked in git)

# Years (edit as needed)
YEARS = list(range(2015, 2025))

# Input file naming (edit if yours differs)
# Example: 2018_listing.csv
LISTING_CSV_PATTERN = "{year}_listing.csv"

# Reference mapping files (place these in /references)
HOSPITAL_MAP_XLSX = REF_DIR / "Hos_Code-15-Jan.csv"
DRUG_MAP_XLSX     = REF_DIR / "Drug_dict.csv"
ORG_MAP_XLSX      = REF_DIR / "mapped_org.csv"

# Common output naming
DASH_CSV_PATTERN = "processed_{year}.csv"
DASH_PARQUET_PATTERN = "processed_{year}.parquet"
COMBINED_PARQUET_NAME = "AllYears_processed.parquet"

# Columns expected in listing (adjust if needed)
COL_LAB = "LABORATORY"
COL_HOS_NAME = "X_HOS_NAME"
COL_WARD_TYPE = "WARD_TYPE"
COL_AGE = "AGE"
COL_SPEC_DATE = "SPEC_DATE"
COL_ORGANISM = "ORGANISM"
COL_ORG_WHO = "ORGANISM_WHO"