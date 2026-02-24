import pandas as pd
import numpy as np
import os

# 1. กำหนด Path (ปรับให้ตรงกับเครื่องของคุณ)
INPUT_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend/amr-a_baumannii_selected.csv'
OUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend/Spatiotemporal'
os.makedirs(OUT_DIR, exist_ok=True)

def prepare_inla_data():    
    # 2. โหลดข้อมูล
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    
    # แปลงวันที่ และสร้างคอลัมน์ที่จำเป็น
    df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
    df = df.dropna(subset=['spec_date', 'region'])
    
    # ดึงเฉพาะปีอดีต (2015-2024)
    df = df[(df['spec_date'].dt.year >= 2015) & (df['spec_date'].dt.year <= 2024)].copy()
    
    # สร้าง SPEC_DATE (YYYY-MM) และ Year
    df['SPEC_DATE'] = df['spec_date'].dt.strftime('%Y-%m')
    df['Year'] = df['spec_date'].dt.year

    # 3. แปลงคอลัมน์ยาให้อยู่ในรูปแบบ Long Format (Unpivot)
    df_melt = df.melt(
        id_vars=['organism_full', 'region', 'SPEC_DATE', 'Year'],
        value_vars=['imipenem', 'meropenem'],
        var_name='ANTIBIOTIC', 
        value_name='RESULT'
    )
    
    # กรองค่าว่างทิ้ง และกำหนด R=1, อื่นๆ=0
    df_melt = df_melt.dropna(subset=['RESULT'])
    df_melt['is_R'] = df_melt['RESULT'].astype(str).str.upper().apply(lambda x: 1 if x == 'R' else 0)
    
    # 4. รวมกลุ่มคำนวณ R_percent
    agg = df_melt.groupby(['organism_full', 'ANTIBIOTIC', 'region', 'SPEC_DATE', 'Year']).agg(
        n_tested=('is_R', 'count'),
        n_resistant=('is_R', 'sum')
    ).reset_index()
    
    agg['R_percent'] = ((agg['n_resistant'] / agg['n_tested']) * 100).round(2)
    
    agg = agg.rename(columns={'organism_full': 'GROUP_NAME', 'region': 'REGION'})
    agg['ANTIBIOTIC'] = agg['ANTIBIOTIC'].str.capitalize()
    
    data_actual = agg[['GROUP_NAME', 'ANTIBIOTIC', 'REGION', 'SPEC_DATE', 'Year', 'R_percent']]

    # ==========================================
    # 🌟 5. สร้าง Grid ครอบคลุมอดีตและอนาคต (2015 - 2029)
    # ==========================================
    regions = sorted(data_actual['REGION'].unique())
    drugs = ['Imipenem', 'Meropenem']
    dates_full = pd.date_range(start='2015-01-01', end='2029-12-01', freq='MS')
    
    # สร้างตารางเปล่าๆ 13 เขต x 2 ยา x 180 เดือน
    grid = pd.MultiIndex.from_product(
        [['Acinetobacter baumannii'], drugs, regions, dates_full], 
        names=['GROUP_NAME', 'ANTIBIOTIC', 'REGION', 'date_obj']
    ).to_frame(index=False)
    
    grid['SPEC_DATE'] = grid['date_obj'].dt.strftime('%Y-%m')
    grid['Year'] = grid['date_obj'].dt.year
    
    # นำข้อมูลจริงไปแปะลงตาราง Grid (เดือนในอนาคต R_percent จะกลายเป็น NaN อัตโนมัติ)
    final_df = pd.merge(grid, data_actual, on=['GROUP_NAME', 'ANTIBIOTIC', 'REGION', 'SPEC_DATE', 'Year'], how='left')
    
    # ==========================================
    # 6. สร้างฟีเจอร์สำหรับ INLA
    # ==========================================
    final_df['month_num'] = final_df['date_obj'].dt.month
    final_df['month_id'] = ((final_df['Year'] - 2015) * 12) + final_df['month_num'] # 1 ถึง 180
    
    # สเกลค่าเปอร์เซ็นต์ให้อยู่ระหว่าง 0-1 สำหรับ INLA Beta Family (อิงตามโค้ดรุ่นพี่)
    # ถ้าเป็น NaN (อนาคต) ก็ปล่อยเป็น NaN ไว้ INLA จะทำนายให้เอง
    final_df['R_scaled'] = (final_df['R_percent'] + 0.001) / 100
    final_df['R_scaled'] = final_df['R_scaled'].clip(0.0001, 0.9999)
    
    # ลบคอลัมน์วันที่ชั่วคราวออก
    final_df = final_df.drop(columns=['date_obj'])
    
    # เซฟไฟล์
    save_path = os.path.join(OUT_DIR, 'INLA_Prepared_Data.csv')
    final_df.to_csv(save_path, index=False)
    
    print("\n✅ หน้าตาข้อมูลที่ได้ (เหมือนรุ่นพี่):")
    print(final_df[['GROUP_NAME', 'ANTIBIOTIC', 'REGION', 'SPEC_DATE', 'Year', 'R_percent']].head())
    print(f"\n📁 บันทึกไฟล์สำเร็จ: {save_path}")
    print("👉 ให้นำไฟล์นี้ไปรันต่อในโปรแกรม RStudio เพื่อทำ Spatiotemporal Model ครับ!")

if __name__ == "__main__":
    prepare_inla_data()