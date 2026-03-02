import pandas as pd
import numpy as np
import os
import glob
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
import warnings

# ปิดการแจ้งเตือนหยุมหยิม
warnings.simplefilter("ignore")

# 1. กำหนดโฟลเดอร์ที่มีไฟล์ CSV ทั้งหมด
folder_path = r"C:\AMR_Thesis\MDR\model\By_specimen"

def calculate_metrics(series):
    """ฟังก์ชันคำนวณ ACF Lag 12 และ Seasonal Strength"""
    if len(series.dropna()) < 24:
        return np.nan, np.nan
    
    # --- ACF at Lag 12 ---
    acf_values = acf(series, nlags=12, fft=True)
    acf_lag12 = acf_values[12] if len(acf_values) > 12 else np.nan
    
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

# 2. ค้นหาไฟล์ CSV ทั้งหมดในโฟลเดอร์
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# ลิสต์สำหรับเก็บผลลัพธ์ของทุกเชื้อและทุกประเภทสิ่งส่งตรวจ (Specimen)
all_results = []

# สร้าง Time Index ปี 2015 - 2024 ไว้รอก่อนเลย
all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
full_index = pd.DataFrame({
    'year': all_months.year,
    'month': all_months.month
})

# เปลี่ยนชื่อคอลัมน์ตอน Print ให้เป็น Specimen Type
print(f"{'Bacteria':<20} | {'Specimen Type':<15} | {'Drug Class':<35} | {'ACF Lag12':<10} | {'Seasonal Strength':<15}")
print("-" * 105)

# 3. วนลูปอ่านแต่ละไฟล์
for file_path in csv_files:
    filename = os.path.basename(file_path)
    bacteria_name = filename.replace('.csv', '').replace('_', ' ').strip().capitalize()
    
    try:
        df = pd.read_csv(file_path)
        
        # ตรวจหาคอลัมน์ spec_type (รองรับทั้งตัวเล็กตัวใหญ่)
        spec_col = next((col for col in df.columns if col.lower() == 'spec_type'), None)
        
        if not spec_col:
            print(f"⚠️ ไม่พบคอลัมน์ 'spec_type' ในไฟล์ {filename} ข้ามการทำงานไฟล์นี้")
            continue
            
        # หาประเภทสิ่งส่งตรวจทั้งหมดที่มีในไฟล์นี้และไม่เป็นค่าว่าง
        unique_specs = df[spec_col].dropna().unique()
        
        # วนลูปประมวลผลทีละ Specimen Type
        for spec in unique_specs:
            # กรองข้อมูลเอาเฉพาะของ Specimen ปัจจุบัน
            spec_df = df[df[spec_col] == spec]
            
            # 4. แปลงเป็น Wide Format
            pivot_df = spec_df.pivot_table(
                index=['year', 'month'], 
                columns='Resistant_Drug_Classes', 
                values='percentage'
            )
            
            # รวมข้อมูลเข้ากับ Index สมบูรณ์และจัดการค่าแหว่ง
            final_df = pd.merge(full_index, pivot_df.reset_index(), on=['year', 'month'], how='left')
            final_df = final_df.interpolate(method='linear').fillna(0)
            
            # ดึงชื่อกลุ่มยาทั้งหมดของเซ็ตนี้
            drug_classes = [col for col in final_df.columns if col not in ['year', 'month']]
            
            # 5. วนลูปคำนวณแต่ละกลุ่มยาในเชื้อและ Specimen นั้นๆ
            for col in drug_classes:
                acf_12, s_strength = calculate_metrics(final_df[col])
                
                # จัดรูปแบบข้อความสำหรับแสดงผล
                b_display = (bacteria_name[:17] + '..') if len(bacteria_name) > 20 else bacteria_name
                sp_display = (str(spec)[:12] + '..') if len(str(spec)) > 15 else str(spec)
                d_display = (col[:32] + '..') if len(col) > 35 else col
                
                acf_display = f"{acf_12:>9.4f}" if not np.isnan(acf_12) else f"{'N/A':>9}"
                s_display = f"{s_strength:>15.4f}" if not np.isnan(s_strength) else f"{'N/A':>15}"
                
                print(f"{b_display:<20} | {sp_display:<15} | {d_display:<35} | {acf_display} | {s_display}")
                
                # เก็บผลลัพธ์ใส่ลิสต์รวม (เปลี่ยนชื่อ Key เป็น Spec_Type)
                all_results.append({
                    'Bacteria': bacteria_name,
                    'Spec_Type': spec,
                    'Resistant_Drug_Classes': col,
                    'ACF_Lag12': acf_12,
                    'Seasonal_Strength': s_strength,
                    'Has_Seasonality_ACF': acf_12 >= 0.5 if not np.isnan(acf_12) else False,
                    'Has_Seasonality_Strength': s_strength >= 0.3 if not np.isnan(s_strength) else False
                })
            
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์ {filename}: {e}")

# 6. บันทึกผลสรุปทั้งหมด
if all_results:
    summary_df = pd.DataFrame(all_results)
    
    save_dir = os.path.dirname(folder_path)
    # เปลี่ยนชื่อไฟล์ผลลัพธ์ให้เป็นคำว่า spec แทน ward
    summary_file = os.path.join(save_dir, "Spec_type_seasonality_summary.csv")
    
    summary_df.to_csv(summary_file, index=False)
    
    print("-" * 105)
    print(f"✅ ประมวลผลเสร็จสิ้น! บันทึกตารางรวมผลลัพธ์ไว้ที่: {summary_file}")
else:
    print("ไม่พบข้อมูลให้ประมวลผล โปรดตรวจสอบข้อมูลในไฟล์อีกครั้งครับ")