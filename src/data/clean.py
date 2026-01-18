from __future__ import annotations
import pandas as pd
from typing import Dict, Any

def assign_age_group(age: Any) -> str:
    """Age grouping logic copied from your notebook (robust to NaN/str)."""
    try:
        if pd.isna(age):
            return "Unknown"
        age = float(age)
    except Exception:
        return "Unknown"

    if age <= 0 or age > 110:
        return "Unknown"

    # Special rule
    if age == 1:
        return "<1"
    elif 2 <= age <= 4:
        return "1-4"
    elif 5 <= age <= 14:
        return "5-14"
    elif 15 <= age <= 24:
        return "15-24"
    elif 25 <= age <= 34:
        return "25-34"
    elif 35 <= age <= 44:
        return "35-44"
    elif 45 <= age <= 54:
        return "45-54"
    elif 55 <= age <= 64:
        return "55-64"
    elif 65 <= age <= 74:
        return "65-74"
    elif 75 <= age <= 84:
        return "75-84"
    else:
        return "85+"

def normalize_ward_type(series: pd.Series) -> pd.Series:
    ward_map = {
        "out": "out",
        "Out": "out",
        "ipd": "in",
        "opd": "out",
        "in": "in",
        "In": "in",
        "Oth": "oth",
        "OTH": "oth",
    }
    return series.apply(lambda x: ward_map.get(x, x))

def rename_drug_columns(columns: list[str], drug_full_by_whonet5: Dict[str, str]) -> Dict[str, str]:
    """Return a rename dict for df.rename based on WHONET 5-char drug prefix."""
    def rename_one(col: str) -> str:
        if "_" not in col:
            return col
        prefix = col.split("_")[0]
        return drug_full_by_whonet5.get(prefix, col)

    return {c: rename_one(c) for c in columns}

def clean_listing_df(
    df: pd.DataFrame,
    drug_full_by_whonet5: Dict[str, str],
    organism_clean_by_org: Dict[str, str],
    col_ward_type: str = "WARD_TYPE",
    col_age: str = "AGE",
    col_organism: str = "ORGANISM",
) -> pd.DataFrame:
    """Clean a single-year listing dataframe."""
    df = df.copy()

    # Normalize ward type
    if col_ward_type in df.columns:
        df[col_ward_type] = normalize_ward_type(df[col_ward_type])

    # Age group
    if col_age in df.columns:
        df["AGE_GROUP"] = df[col_age].apply(assign_age_group)

    # Rename drug columns
    df = df.rename(columns=rename_drug_columns(list(df.columns), drug_full_by_whonet5))

    # Organism mapping
    if col_organism in df.columns:
        df["ORGANISM_FULL"] = df[col_organism].map(organism_clean_by_org)

    return df
