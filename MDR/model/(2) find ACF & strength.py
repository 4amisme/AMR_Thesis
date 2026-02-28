import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
import warnings

# ปิดการแจ้งเตือนหยุมหยิม
warnings.simplefilter("ignore")

# 1. กำหนดเส้นทางไฟล์
file_path = os.path.join("MDR", "model", "a_baumannii_bl.csv")

def calculate_metrics(series):
    """ฟังก์ชันคำนวณ ACF Lag 12 และ Seasonal Strength"""
    # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่ (STL ต้องการอย่างน้อย 2 รอบฤดูกาล คือ 24 จุด)
    if len(series.dropna()) < 24:
        return np.nan, np.nan
    
    # --- ACF at Lag 12 ---
    # nlags ต้องครอบคลุมถึง 12
    acf_values = acf(series, nlags=12, fft=True)
    acf_lag12 = acf_values[12] if len(acf_values) > 12 else np.nan
    
    # --- Seasonal Strength ---
    try:
        # ใช้ STL Decomposition (period=12 สำหรับข้อมูลรายเดือน)
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

try:
    # 2. อ่านข้อมูล
    df = pd.read_csv(file_path)

    # 3. แปลงเป็น Wide Format
    # เราใช้ pivot เพื่อกาง Resistant_Drug_Classes ออกเป็นคอลัมน์
    pivot_df = df.pivot_table(
        index=['year', 'month'], 
        columns='Resistant_Drug_Classes', 
        values='percentage'
    )

    # 4. สร้าง Time Index ให้ครบทุกเดือน (2015 - 2024)
    # เพื่อป้องกันปัญหาเวลาคำนวณ ACF แล้วข้อมูลไม่ต่อเนื่อง
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_index = pd.DataFrame({
        'year': all_months.year,
        'month': all_months.month
    })

    # รวมข้อมูลที่มีเข้ากับ Index ที่สมบูรณ์
    final_df = pd.merge(full_index, pivot_df.reset_index(), on=['year', 'month'], how='left')
    
    # [จุดที่แก้ไข] ลากเส้นเชื่อมข้อมูลที่แหว่งหายไป (Interpolate) เพื่อไม่ให้สถิติเพี้ยน
    # และใช้ fillna(0) ปิดท้ายเฉพาะเดือนแรกสุดที่ไม่มีข้อมูลให้อ้างอิง
    final_df = final_df.interpolate(method='linear').fillna(0)

    # 5. วนลูปคำนวณค่าของแต่ละกลุ่มยา (Columns)
    results = []
    
    # ข้ามคอลัมน์ year และ month เพื่อให้เหลือแค่ชื่อกลุ่มยา
    drug_classes = [col for col in final_df.columns if col not in ['year', 'month']]

    print(f"{'Drug Class':<50} | {'ACF Lag12':<10} | {'Seasonal Strength':<15}")
    print("-" * 85)

    for col in drug_classes:
        acf_12, s_strength = calculate_metrics(final_df[col])
        
        # ตัดข้อความชื่อกลุ่มยาให้สั้นลงถ้าจำเป็นเพื่อความสวยงามในการแสดงผล
        display_name = (col[:47] + '..') if len(col) > 50 else col
        
        # ป้องกันกรณีคำนวณไม่ได้ (np.nan) ให้แสดงผลสวยๆ
        acf_display = f"{acf_12:>9.4f}" if not np.isnan(acf_12) else f"{'N/A':>9}"
        s_display = f"{s_strength:>15.4f}" if not np.isnan(s_strength) else f"{'N/A':>15}"
        
        print(f"{display_name:<50} | {acf_display} | {s_display}")
        
        # เก็บผลลัพธ์ใส่ list
        results.append({
            'Resistant_Drug_Classes': col,
            'ACF_Lag12': acf_12,
            'Seasonal_Strength': s_strength,
            'Has_Seasonality_ACF': acf_12 >= 0.5 if not np.isnan(acf_12) else False,
            'Has_Seasonality_Strength': s_strength >= 0.3 if not np.isnan(s_strength) else False
        })

    # 6. บันทึกผลสรุป
    summary_df = pd.DataFrame(results)
    summary_file = os.path.join("MDR", "model", "acinetobacter_seasonality_summary.csv")
    
    # ตรวจสอบว่ามีโฟลเดอร์ MDR/model หรือยัง ถ้ายังไม่มีให้สร้างก่อน
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    summary_df.to_csv(summary_file, index=False)
    
    print("-" * 85)
    print(f"บันทึกผลสรุปค่าสถิติไว้ที่: {summary_file}")

except FileNotFoundError:
    print(f"ไม่พบไฟล์: {file_path} กรุณาตรวจสอบ Path อีกครั้ง")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")