from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class Mappings:
    """Container for mapping dictionaries used in cleaning."""
    hos_name_by_lab_code: dict
    drug_full_by_whonet5: dict
    organism_clean_by_org: dict

def load_mappings(
    hospital_xlsx: Path,
    drug_xlsx: Path,
    org_xlsx: Path,

) -> Mappings:
    """Load mapping reference tables from Excel files."""
    hos_map = pd.read_excel(hospital_xlsx)
    # Your notebook mapped LABORATORY (lab_code) -> hos_name
    hos_name_by_lab_code = dict(zip(hos_map["Code"], hos_map["โรงพยาบาล"]))

    drug_map_df = pd.read_excel(drug_xlsx)
    # WHON5_CODE -> ANTIBIOTIC
    drug_full_by_whonet5 = dict(zip(drug_map_df["Code"], drug_map_df["Antibiotic"]))

    org_map = pd.read_excel(org_xlsx)
    # ORG -> ORG_CLEAN
    organism_clean_by_org = dict(zip(org_map["Code_Org"], org_map["Organism"]))

    return Mappings(
        hos_name_by_lab_code=hos_name_by_lab_code,
        drug_full_by_whonet5=drug_full_by_whonet5,
        organism_clean_by_org=organism_clean_by_org,
    )
