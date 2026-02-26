import pandas as pd
import os
import numpy as np

INPUT_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend/amr-a_baumannii_selected.csv'
BASE_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend'

def step2_prepare_monthly():    
    folders = ['All data', 'by ward', 'by specimen']
    for folder in folders:
        for sub in ['Data', 'Seasonality', 'Forecasts']:
            os.makedirs(os.path.join(BASE_DIR, folder, sub), exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    if 'spec_date' not in df.columns:
        print("ไม่พบคอลัมน์ 'spec_date'")
        return
    df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
    df = df.dropna(subset=['spec_date'])
    df = df[(df['spec_date'].dt.year >= 2015) & (df['spec_date'].dt.year <= 2024)].copy()
    
    df['x_year'] = df['spec_date'].dt.year
    df['x_month'] = df['spec_date'].dt.month
    df['date'] = pd.to_datetime(df['x_year'].astype(str) + '-' + df['x_month'].astype(str).str.zfill(2) + '-01')

    # %R ของ Imipenem และ Meropenem
    def calculate_percent_r(data, group_cols):
        results = []
        for drug in ['imipenem', 'meropenem']:
            if drug not in data.columns:
                continue
            
            df_drug = data.dropna(subset=[drug]).copy()
            df_drug['is_resistant'] = df_drug[drug].astype(str).str.upper().apply(lambda x: 1 if x == 'R' else 0)
            
            agg_df = df_drug.groupby(group_cols).agg(
                n_tested=('is_resistant', 'count'),
                n_resistant=('is_resistant', 'sum')
            ).reset_index()
            agg_df['drug'] = drug.capitalize()
            agg_df['percent_R'] = ((agg_df['n_resistant'] / agg_df['n_tested']) * 100).round(2)
            results.append(agg_df)
            
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
    
    overall_df = calculate_percent_r(df, ['date'])
    overall_df.to_csv(os.path.join(BASE_DIR, 'All data', 'Data', 'monthly_overall.csv'), index=False)
    print("All data/Data/monthly_overall.csv")
    
    df_ward = df[df['ward_type'].isin(['icu', 'in', 'out'])].copy()
    ward_df = calculate_percent_r(df_ward, ['date', 'ward_type'])
    ward_df.to_csv(os.path.join(BASE_DIR, 'by ward', 'Data', 'monthly_ward.csv'), index=False)
    print("by ward/Data/monthly_ward.csv")
    
    df_spec = df.dropna(subset=['spec_group']).copy()
    spec_df = calculate_percent_r(df_spec, ['date', 'spec_group'])
    spec_df.to_csv(os.path.join(BASE_DIR, 'by specimen', 'Data', 'monthly_specimen.csv'), index=False)
    print("by specimen/Data/monthly_specimen.csv")

if __name__ == "__main__":
    step2_prepare_monthly()