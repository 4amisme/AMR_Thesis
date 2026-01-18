from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


def _norm_text(x) -> str:
    """Normalize Thai/EN hospital names for matching."""
    if pd.isna(x):
        return ""

    s = str(x).strip()

    # ลบคำว่า โรงพยาบาล
    s = s.replace("โรงพยาบาล", "")

    # collapse whitespace
    s = " ".join(s.split())

    return s

@dataclass(frozen=True)
class Mappings:
    """Container for mapping dictionaries used in cleaning."""
    hos_name_by_lab_code: dict
    hos_code_by_hos_name: dict
    hos_name_by_hos_code: dict
    drug_full_by_whonet5: dict
    organism_clean_by_org: dict

def load_mappings(
    hospital_xlsx: Path,
    drug_xlsx: Path,
    org_xlsx: Path,
) -> Mappings:
    """Load mapping reference tables from reference files."""

    hos_map = pd.read_csv(hospital_xlsx, dtype=str)

    # LABORATORY code -> hospital name
    hos_name_by_lab_code = dict(
        zip(hos_map["Code"], hos_map["โรงพยาบาล"])
    )

    # hospital name -> hospital code (แก้ X_HOS_CODE NA)
    hos_code_by_hos_name = (
        hos_map
        .dropna(subset=["โรงพยาบาล", "Code"])
        .assign(_hos_key=hos_map["โรงพยาบาล"].map(_norm_text))
        .drop_duplicates(subset=["_hos_key"])
        .set_index("_hos_key")["Code"]
        .to_dict()
    )

    # hospital code -> hospital name
    hos_name_by_hos_code = (
        hos_map
        .dropna(subset=["Code", "โรงพยาบาล"])
        .drop_duplicates(subset=["Code"])
        .set_index("Code")["โรงพยาบาล"]
        .to_dict()
    )

    # ---------- Drug reference ----------
    drug_map_df = pd.read_csv(drug_xlsx)
    drug_full_by_whonet5 = dict(
        zip(drug_map_df["Code"], drug_map_df["Antibiotic"])
    )

    # ---------- Organism reference ----------
    org_map = pd.read_csv(org_xlsx)
    organism_clean_by_org = dict(
        zip(org_map["Code_Org"], org_map["Organism"])
    )

    return Mappings(
        hos_name_by_lab_code=hos_name_by_lab_code,
        hos_code_by_hos_name=hos_code_by_hos_name,
        hos_name_by_hos_code=hos_name_by_hos_code,
        drug_full_by_whonet5=drug_full_by_whonet5,
        organism_clean_by_org=organism_clean_by_org,
    )
