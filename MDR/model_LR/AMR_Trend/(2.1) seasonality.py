import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

BASE_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend'

def calculate_metrics(series):
    if len(series.dropna()) < 24:
        return np.nan, np.nan
    
    # --- ACF at Lag 12 ---
    try:
        acf_values = acf(series, nlags=12, fft=True)
        acf_lag12 = acf_values[12] if len(acf_values) > 12 else np.nan
    except:
        acf_lag12 = np.nan
    
    # --- Seasonal Strength ---
    try:
        res = STL(series, period=12, robust=True).fit()
        var_resid = np.var(res.resid)
        var_seasonal_resid = np.var(res.seasonal + res.resid)
        
        if var_seasonal_resid == 0:
            seasonal_strength = 0
        else:
            seasonal_strength = max(0, 1 - (var_resid / var_seasonal_resid))
    except:
        seasonal_strength = np.nan
        
    return acf_lag12, seasonal_strength

def process_seasonality(df, category_folder, group_col=None):
    df['date'] = pd.to_datetime(df['date'])
    full_idx = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    
    groups = [None] if group_col is None else df[group_col].dropna().unique()
    summary_results = []
    
    for g in groups:
        for drug in ['vancomycin']:
            temp = df[df['drug'] == drug.capitalize()]
            if group_col: 
                temp = temp[temp[group_col] == g]
            
            if temp.empty:
                continue
            series = temp.set_index('date')['percent_R']
            series = series.reindex(full_idx).interpolate(method='linear').fillna(0)
            acf_12, s_strength = calculate_metrics(series)
            has_acf = acf_12 >= 0.5 if not np.isnan(acf_12) else False
            has_strength = s_strength >= 0.3 if not np.isnan(s_strength) else False
            
            summary_results.append({
                'Category': category_folder,
                'Group': g if g else 'Overall',
                'Drug': drug,
                'ACF_Lag12': round(acf_12, 4) if not np.isnan(acf_12) else np.nan,
                'Seasonal_Strength': round(s_strength, 4) if not np.isnan(s_strength) else np.nan,
                'Has_Seasonality_ACF': has_acf,
                'Has_Seasonality_Strength': has_strength
            })
            
    return summary_results

def step3_check_seasonality():
    all_summaries = []
    
    file_all = os.path.join(BASE_DIR, 'All data', 'Data', 'efa_monthly_overall.csv')
    if os.path.exists(file_all):
        df_all = pd.read_csv(file_all)
        all_summaries.extend(process_seasonality(df_all, 'All data', None))
    
    file_ward = os.path.join(BASE_DIR,'by ward', 'Data', 'efa_monthly_ward.csv')
    if os.path.exists(file_ward):
        df_ward = pd.read_csv(file_ward)
        all_summaries.extend(process_seasonality(df_ward, 'by ward', 'ward_type'))
    
    file_spec = os.path.join(BASE_DIR, 'by specimen', 'Data', 'efa_monthly_specimen.csv')
    if os.path.exists(file_spec):
        df_spec = pd.read_csv(file_spec)
        df_spec = df_spec[df_spec['spec_group'] != 'other']
        all_summaries.extend(process_seasonality(df_spec, 'by specimen', 'spec_group'))

    # Seasonality Strength
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        save_csv_path = os.path.join(BASE_DIR, 'efa_Seasonality_Strength_Summary.csv')
        summary_df.to_csv(save_csv_path, index=False)
        
        print(f" save ไฟล์ที่:\n   {save_csv_path}")
    else:
        print("ไม่พบข้อมูล")

if __name__ == "__main__":
    step3_check_seasonality()