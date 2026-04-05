import pandas as pd
import numpy as np
import os
import glob
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
import warnings

# ปิดการแจ้งเตือน
warnings.simplefilter("ignore")

# 1. กำหนด Path (โฟลเดอร์ Ward ที่มีโฟลเดอร์ย่อยของเชื้ออยู่ข้างใน)
folder_path = r"C:\AMR_Thesis\MDR\model_for_1_Drug\Ward"

def calculate_metrics(series):
    """คำนวณ ACF Lag 12 และ Seasonal Strength (STL)"""
    # ข้อมูลต้องมีอย่างน้อย 24 เดือน (2 ปี) เพื่อให้เห็นวงจรฤดูกาล
    if len(series.dropna()) < 24:
        return np.nan, np.nan
    
    try:
        # 1. ACF Lag 12: วัดความสัมพันธ์ของข้อมูลเดือนเดียวกันในแต่ละปี
        acf_values = acf(series, nlags=12, fft=True)
        acf_lag12 = acf_values[12] if len(acf_values) > 12 else np.nan
        
        # 2. Seasonal Strength: คำนวณความแรงขององค์ประกอบฤดูกาลจาก STL
        res = STL(series, period=12, robust=True).fit()
        var_resid = np.var(res.resid)
        var_seasonal_resid = np.var(res.seasonal + res.resid)
        
        strength = max(0, 1 - (var_resid / var_seasonal_resid)) if var_seasonal_resid != 0 else 0
        return acf_lag12, strength
    except:
        return np.nan, np.nan

# 2. ค้นหาไฟล์ CSV ทั้งหมดในโฟลเดอร์ย่อย (Recursive)
# จะค้นหาไฟล์ เช่น a_baumannii_icu.csv ในทุกโฟลเดอร์ย่อย
csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)

# สร้าง Index เวลามาตรฐาน 120 เดือน (2015-2024)
all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
full_index = pd.DataFrame({'year': all_months.year, 'month': all_months.month})

all_results = []

print(f"🔎 พบไฟล์ข้อมูล Ward ทั้งหมด {len(csv_files)} ไฟล์")
print("-" * 115)
print(f"{'Bacteria':<20} | {'Ward':<8} | {'Drug Name':<20} | {'ACF L12':<8} | {'Seas. Strength'}")
print("-" * 115)

# 3. วนลูปประมวลผลรายไฟล์
for file_path in csv_files:
    filename = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        
        if df.empty: continue
        
        # ดึงข้อมูล Organism และ Ward จากคอลัมน์ในไฟล์
        bacteria = df['organism_full'].iloc[0]
        ward_type = df['ward_type'].iloc[0]
        drug_col = 'resistant_drug_name'
        
        # จัดการเคสที่ไม่ดื้อยา (Sensitive) ให้มีชื่อเรียก
        df[drug_col] = df[drug_col].fillna('Sensitive/None')

        # 4. แปลงข้อมูลเป็นรูปแบบ Time Series
        monthly_avg = df.groupby(['year', 'month', drug_col])['percentage'].mean().reset_index()
        pivot_df = monthly_avg.pivot_table(index=['year', 'month'], columns=drug_col, values='percentage')
        
        # เชื่อมกับ Index เวลามาตรฐาน และเติมค่าที่ขาดหายด้วยการประมาณค่า (Interpolate)
        final_df = pd.merge(full_index, pivot_df.reset_index(), on=['year', 'month'], how='left')
        final_df = final_df.interpolate(method='linear').fillna(0)
        
        # ดึงรายชื่อยาทุกชนิดที่มีในไฟล์ Ward นี้
        drug_names = [col for col in final_df.columns if col not in ['year', 'month']]
        
        for drug in drug_names:
            acf_12, s_strength = calculate_metrics(final_df[drug])
            
            # การแสดงผลทางหน้าจอ
            b_disp = (bacteria[:18] + "..") if len(bacteria) > 20 else bacteria
            w_disp = str(ward_type).upper()
            d_disp = (str(drug)[:18] + "..") if len(str(drug)) > 20 else str(drug)
            
            acf_p = f"{acf_12:>8.3f}" if not np.isnan(acf_12) else "   N/A"
            str_p = f"{s_strength:>13.3f}" if not np.isnan(s_strength) else "      N/A"
            
            print(f"{b_disp:<20} | {w_disp:<8} | {d_disp:<20} | {acf_p} | {str_p}")
            
            all_results.append({
                'Bacteria': bacteria,
                'Ward_Type': ward_type,
                'Drug_Name': drug,
                'ACF_Lag12': acf_12,
                'Seasonal_Strength': s_strength,
                'Has_Seasonality': (acf_12 >= 0.4 or s_strength >= 0.3)
            })
            
    except Exception as e:
        print(f"❌ Error ในไฟล์ {filename}: {e}")

# 5. บันทึกตารางสรุปผล
if all_results:
    summary_df = pd.DataFrame(all_results)
    output_file = os.path.join(folder_path, "Ward_Seasonality.csv")
    summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("-" * 115)
    print(f"✅ บันทึกตารางสรุปผลราย Ward เรียบร้อย: {output_file}")