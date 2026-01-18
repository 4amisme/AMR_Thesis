from __future__ import annotations
from pathlib import Path
import pandas as pd

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_year_output(df: pd.DataFrame, out_dir: Path, year: int, fmt: str = "parquet") -> Path:
    """Write cleaned single-year dataset to processed folder."""
    ensure_dir(out_dir)
    if fmt == "csv":
        out_path = out_dir / f"no_drug_map_{year}.csv"
        df.to_csv(out_path, index=False)
        return out_path
    elif fmt == "parquet":
        out_path = out_dir / f"no_drug_map_{year}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path
    else:
        raise ValueError("fmt must be 'csv' or 'parquet'")

def combine_years(processed_dir: Path, years: list[int], fmt: str = "parquet") -> Path:
    """Combine per-year outputs into a single file (for modeling convenience)."""
    dfs = []
    for y in years:
        p_parq = processed_dir / f"no_drug_map_{y}.parquet"
        p_csv = processed_dir / f"no_drug_map_{y}.csv"
        if p_parq.exists():
            dfs.append(pd.read_parquet(p_parq))
        elif p_csv.exists():
            dfs.append(pd.read_csv(p_csv))
        else:
            continue

    if not dfs:
        raise FileNotFoundError("No yearly processed files found to combine.")

    df_all = pd.concat(dfs, ignore_index=True)

    ensure_dir(processed_dir)
    if fmt == "csv":
        out = processed_dir / "AllYears_no_drug_map.csv"
        df_all.to_csv(out, index=False)
        return out
    else:
        out = processed_dir / "AllYears_no_drug_map.parquet"
        df_all.to_parquet(out, index=False)
        return out
