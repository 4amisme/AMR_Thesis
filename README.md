This folder contains a refactored project scaffold for Pam's AMR_Thesis.

Where to put files:
- Place yearly WHONET-exported listing CSVs in: data/raw/{year}_listing.csv
- Place reference mapping excels in: references/
    - Hos_THA.xlsx (sheet Hos_THA)
    - Drug.xlsx (sheet Map Drug)
    - Organism.xlsx (sheet ORGLIST)

Run:
- python scripts/run_preprocess.py

Outputs:
- data/processed/Dash_{year}.parquet
- data/processed/AllYears_Dash.parquet


