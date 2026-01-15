from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_listing_csv(path: Path) -> pd.DataFrame:
    """Read a WHONET-exported listing CSV for a single year."""
    return pd.read_csv(path)

def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely missing."""
    total_rows = df.shape[0]
    cols_all_missing = df.columns[df.isna().sum() == total_rows]
    return df.drop(columns=list(cols_all_missing))
