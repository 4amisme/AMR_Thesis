from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.config import MDR_DIR, ARTIFACTS_DIR

DEFAULT_INPUT_NAME = "AllYears_Standard_Checked_Selected.csv"

TARGET_ORGANISMS: list[str] = [
    "Acinetobacter baumannii",
    "Citrobacter freundii",
    "Enterobacter cloacae",
    "Enterobacter spp.",
    "Enterococcus faecalis",
    "Enterococcus faecium",
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Klebsiella spp.",
    "Proteus mirabilis",
    "Proteus vulgaris",
    "Pseudomonas aeruginosa",
    "Serratia marcescens",
]


def _slugify(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_\-\.]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s or "unknown"


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"organism_full", "std_status", "std_missing_list"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["organism_full"] = df["organism_full"].astype("string").str.strip()
    df["std_status"] = df["std_status"].astype("string").str.strip()
    df["std_missing_list"] = df["std_missing_list"].astype("string").str.strip()

    # 1) organism filter
    m_org = df["organism_full"].isin(TARGET_ORGANISMS)

    # 2) status filter
    m_missing0 = df["std_status"].eq("Complete")
    m_missing1_lipo = df["std_status"].eq("Missing 1") & df["std_missing_list"].eq("Lipopeptides")
    m_status = m_missing0 | m_missing1_lipo

    return df.loc[m_org & m_status].copy()


def export_by_organism(df_filtered: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for org, sub in df_filtered.groupby("organism_full", dropna=False):
        org_str = str(org) if org is not pd.NA else "Unknown"
        out_path = out_dir / f"{_slugify(org_str)}.csv"

        sub.to_csv(out_path, index=False, encoding="utf-8-sig")
        summary_rows.append({"organism_full": org_str, "n_rows": int(len(sub)), "output_file": str(out_path)})

    summary = pd.DataFrame(summary_rows).sort_values(["n_rows", "organism_full"], ascending=[False, True])
    summary.to_csv(out_dir / "_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter isolates and export per organism (MDR_non-lipo).")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_NAME,
        help="CSV filename (relative to MDR_DIR) or full path.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = MDR_DIR / in_path
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_dir = ARTIFACTS_DIR / "MDR_non-lipo"

    print(f"[INFO] Input:  {in_path}")
    print(f"[INFO] Output: {out_dir}")

    df = pd.read_csv(in_path, low_memory=False)
    print(f"[INFO] Loaded rows: {len(df):,}")

    df_filtered = filter_dataframe(df)
    print(f"[INFO] Filtered rows: {len(df_filtered):,}")

    summary = export_by_organism(df_filtered, out_dir)
    print(f"[INFO] Exported {len(summary):,} organism files + _summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
