import pandas as pd
import os
import numpy as np

INPUT_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/data/MDR/AllYears_DrugClass_tested.csv'
OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR'

def step1_extract():
    
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df.columns = df.columns.str.lower()
    
    df = df[(df['x_year'] >= 2015) & (df['x_year'] <= 2024)].copy()
    
    # เลือกเฉพาะเชื้อ A. baumannii
    df_aba = df[df['organism_full'].str.contains('Acinetobacter baumannii', case=False, na=False)].copy()
    

    df_aba['ward_type'] = df_aba['ward_type'].astype(str).str.lower().str.strip()
    df_aba['ward_type'] = df_aba['ward_type'].replace(['nan', 'none', '', 'null'], np.nan)
    
    df_aba['x_spec_ful'] = df_aba['x_spec_ful'].astype(str).str.strip()
    df_aba['x_spec_ful'] = df_aba['x_spec_ful'].replace(['nan', 'none', '', 'null'], np.nan)
    
    valid_specs = df_aba.dropna(subset=['x_spec_ful'])
    top3 = valid_specs['x_spec_ful'].value_counts().nlargest(3).index.tolist()
    print(f"📌 Top 3 Specimens: {top3}")
    
    df_aba['spec_group'] = df_aba['x_spec_ful'].apply(
        lambda x: x if x in top3 else ('other' if pd.notnull(x) else np.nan)
    )
    
    cols_to_keep = [
        'organism_full', 'x_year', 'region', 
        'ward_type', 'x_spec_ful', 'spec_group'
    ]
    df_final = df_aba[cols_to_keep]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(os.path.join(OUTPUT_DIR, 'a_baumannii_selected.csv'), index=False)
    print("✅ Step 1 Complete: Saved selected data.")

if __name__ == "__main__":
    step1_extract()