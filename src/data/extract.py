from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_listing_csv(path: Path) -> pd.DataFrame:
    """Read a WHONET-exported listing CSV for a single year."""
    return pd.read_csv(path, low_memory=False, dtype=str)