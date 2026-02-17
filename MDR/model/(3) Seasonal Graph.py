import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import STL

# 1. ตั้งค่าเส้นทางไฟล์
file_path = os.path.join("MDR", "model", "acinetobacter_baumannii.csv")

try:
    # 2. อ่านข้อมูล
    df = pd.read_csv(file_path)

    # 3. แปลงเป็น Wide Format
    pivot_df = df.pivot_table(
        index=['year', 'month'], 
        columns='Resistant_Drug_Classes', 
        values='percentage'
    )

    # 4. สร้าง Time Index ให้ครบทุกเดือน (2015-01 ถึง 2024-12) เพื่อความต่อเนื่องของกราฟ
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_index = pd.DataFrame({
        'year': all_months.year,
        'month': all_months.month
    })

    # รวมข้อมูลที่มีเข้ากับ Index ที่สมบูรณ์ และเติมเดือนที่ไม่มีข้อมูลด้วย 0
    final_df = pd.merge(full_index, pivot_df.reset_index(), on=['year', 'month'], how='left')
    final_df = final_df.fillna(0)

    # 5. ฟังก์ชันสำหรับสร้างกราฟวิเคราะห์ฤดูกาล
    def plot_seasonality_analysis(data_df, drug_class_name, date_series):
        # ดึงข้อมูลของกลุ่มยานั้นๆ ออกมา
        series = data_df[drug_class_name]
        
        # ตรวจสอบว่ามีข้อมูลเพียงพอสำหรับการทำ STL (อย่างน้อย 24 เดือน)
        if len(series.dropna()) < 24:
            print(f"ข้อมูลของ {drug_class_name} มีน้อยเกินไปสำหรับการวิเคราะห์ฤดูกาล")
            return

        # ทำ STL Decomposition
        # period=12 คือรอบ 12 เดือน (1 ปี)
        res = STL(series, period=12, robust=True).fit()
        
        # สร้างพื้นที่วาดกราฟ 3 ช่อง (Actual, Trend, Seasonal)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # กราฟ 1: ข้อมูลจริง (Observed)
        ax1.plot(date_series, series, color='#1f77b4', linewidth=1.5, label='Observed (Actual %)')
        ax1.set_title(f'Time Series Decomposition: {drug_class_name[:80]}...', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # กราฟ 2: แนวโน้ม (Trend)
        ax2.plot(date_series, res.trend, color='#d62728', linewidth=2, label='Trend (Long-term Direction)')
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # กราฟ 3: ฤดูกาล (Seasonal)
        ax3.plot(date_series, res.seasonal, color='#2ca02c', linewidth=1.5, label='Seasonal (Yearly Pattern)')
        ax3.legend(loc='upper left')
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # ตกแต่งแกน X
        plt.xlabel('Year', fontsize=12)
        plt.tight_layout()
        plt.show()

    # 6. เริ่มการ Plot สำหรับทุกกลุ่มยาที่มีในไฟล์ (Top 5)
    drug_classes = pivot_df.columns.tolist()
    
    print(f"พบกลุ่มยาทั้งหมด {len(drug_classes)} กลุ่ม กำลังสร้างกราฟ...")
    
    for cls in drug_classes:
        plot_seasonality_analysis(final_df, cls, all_months)

except FileNotFoundError:
    print(f"ไม่พบไฟล์: {file_path}")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")