import pandas as pd
import numpy as np
import os
import glob
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
import warnings

# ปิดการแจ้งเตือน
warnings.simplefilter("ignore")

# 1. กำหนด Path (อ้างอิงตามรูปภาพของคุณ)
folder_path = r"C:\AMR_Thesis\MDR\model_1Class\All Data"

def calculate_metrics(series):
    """คำนวณ ACF Lag 12 และ Seasonal Strength"""
    # กรองข้อมูลว่าง และเช็คว่ามีข้อมูลอย่างน้อย 24 เดือน
    clean_series = series.dropna()
    if len(clean_series) < 24:
        return np.nan, np.nan
    
    try:
        # ACF Lag 12
        acf_values = acf(series, nlags=12, fft=True)
        acf_lag12 = acf_values[12] if len(acf_values) > 12 else np.nan
        
        # Seasonal Strength จาก STL
        res = STL(series, period=12, robust=True).fit()
        var_resid = np.var(res.resid)
        var_seasonal_resid = np.var(res.seasonal + res.resid)
        
        strength = max(0, 1 - (var_resid / var_seasonal_resid)) if var_seasonal_resid != 0 else 0
        return acf_lag12, strength
    except:
        return np.nan, np.nan

# 2. ค้นหาไฟล์ CSV ทั้งหมดในโฟลเดอร์ All Data
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# สร้าง Time Index มาตรฐาน 2015-2024
all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
full_index = pd.DataFrame({'year': all_months.year, 'month': all_months.month})

all_results = []

print(f"🔎 พบไฟล์ข้อมูลเชื้อทั้งหมด {len(csv_files)} ไฟล์")
print(f"{'Bacteria Name':<30} | {'Drug Class':<30} | {'ACF L12':<8} | {'Seas. Strength'}")
print("-" * 105)

# 3. วนลูปประมวลผลทีละไฟล์ (1 ไฟล์ = 1 เชื้อ)
for file_path in csv_files:
    filename = os.path.basename(file_path)
    
    # ดึงชื่อเชื้อจากชื่อไฟล์ (เช่น escherichia_coli.csv -> Escherichia Coli)
    bacteria_name = filename.replace('.csv', '').replace('_', ' ').title()
    
    try:
        df = pd.read_csv(file_path)
        # ปรับหัวตารางเป็นตัวเล็ก
        df.columns = df.columns.str.strip().str.lower()
        
        # ตรวจสอบคอลัมน์กลุ่มยา (Resistant Drug Classes)
        drug_col = 'resistant_drug_classes'
        if drug_col not in df.columns:
            # ถ้าหาไม่เจอ ให้ลองหาคอลัมน์ที่ใกล้เคียง
            drug_col = [c for c in df.columns if 'drug' in c or 'class' in c][0]

        # 4. แปลงข้อมูลเป็นรูปแบบ Time Series (Wide Format)
        # หาค่าเฉลี่ย Percentage รายเดือน (ในกรณีที่มีข้อมูลซ้ำในเดือนเดียวกัน)
        monthly_avg = df.groupby(['year', 'month', drug_col])['percentage'].mean().reset_index()
        
        pivot_df = monthly_avg.pivot_table(index=['year', 'month'], columns=drug_col, values='percentage')
        
        # Join กับ Index เวลาให้ครบ 120 เดือน
        final_df = pd.merge(full_index, pivot_df.reset_index(), on=['year', 'month'], how='left')
        # เติมค่าว่างเพื่อให้คำนวณสถิติได้
        final_df = final_df.interpolate(method='linear').fillna(0)
        
        # ดึงรายชื่อกลุ่มยาที่มีในไฟล์นี้
        drug_classes = [col for col in final_df.columns if col not in ['year', 'month']]
        
        for drug in drug_classes:
            acf_12, s_strength = calculate_metrics(final_df[drug])
            
            # แสดงผลทางหน้าจอ
            b_disp = (bacteria_name[:28] + "..") if len(bacteria_name) > 30 else bacteria_name
            d_disp = (drug[:28] + "..") if len(drug) > 30 else drug
            
            acf_p = f"{acf_12:>8.3f}" if not np.isnan(acf_12) else "   N/A"
            str_p = f"{s_strength:>13.3f}" if not np.isnan(s_strength) else "      N/A"
            
            print(f"{b_disp:<30} | {d_disp:<30} | {acf_p} | {str_p}")
            
            all_results.append({
                'Bacteria': bacteria_name,
                'Drug_Class': drug,
                'ACF_Lag12': acf_12,
                'Seasonal_Strength': s_strength,
                'Has_Seasonality': (acf_12 >= 0.4 or s_strength >= 0.3)
            })
            
    except Exception as e:
        print(f"❌ Error ในไฟล์ {filename}: {e}")

# 5. บันทึกผลลัพธ์
if all_results:
    summary_df = pd.DataFrame(all_results)
    output_file = os.path.join(folder_path, "Bacteria_Seasonality_Summary.csv")
    summary_df.to_csv(output_file, index=False)
    print("-" * 105)
    print(f"✅ บันทึกตารางสรุปผลเรียบร้อย: {output_file}")