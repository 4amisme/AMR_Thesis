import pandas as pd
import os
import numpy as np

INPUT_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/data/MDR/AllYears_DrugClass_tested.csv'
OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend'

def step1_extract():
    print("⏳ Step 1: Loading and Cleaning Data...")
    
    # 1. โหลดข้อมูลและแปลงชื่อคอลัมน์เป็นตัวพิมพ์เล็ก
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df.columns = df.columns.str.lower()
    
    # 2. กรองปี 2015-2024
    df = df[(df['x_year'] >= 2015) & (df['x_year'] <= 2024)].copy()
    
    # 3. เลือกเฉพาะเชื้อ A. baumannii
    df_aba = df[df['organism_full'].str.contains('Acinetobacter baumannii', case=False, na=False)].copy()
    
    # 4. จัดการ Ward Type (คลีนค่าว่าง และทำเป็นตัวเล็ก)
    df_aba['ward_type'] = df_aba['ward_type'].astype(str).str.lower().str.strip()
    df_aba['ward_type'] = df_aba['ward_type'].replace(['nan', 'none', '', 'null'], np.nan)
    
    # 5. จัดการ Specimen (Top 3 + Other)
    df_aba['x_spec_ful'] = df_aba['x_spec_ful'].astype(str).str.strip()
    df_aba['x_spec_ful'] = df_aba['x_spec_ful'].replace(['nan', 'none', '', 'null'], np.nan)
    
    valid_specs = df_aba.dropna(subset=['x_spec_ful'])
    top3 = valid_specs['x_spec_ful'].value_counts().nlargest(3).index.tolist()
    print(f"📌 Top 3 Specimens: {top3}")
    
    df_aba['spec_group'] = df_aba['x_spec_ful'].apply(
        lambda x: x if x in top3 else ('other' if pd.notnull(x) else np.nan)
    )
    
    # 6. เลือกเฉพาะคอลัมน์ที่ใช้งาน (รวมเดือน/วันที่ และ ผลตรวจยาเตรียมไว้สำหรับ Time Series)
    desired_cols = [
        'organism_full', 'x_year', 'region', 
        'ward_type', 'x_spec_ful', 'spec_group',
        'spec_date', 'x_month', 'imipenem', 'meropenem'
    ]
    
    # เช็คว่ามีคอลัมน์ตาม desired_cols ครบไหม เพื่อป้องกัน Error (ถ้าไม่มีก็ข้ามไป)
    cols_to_keep = [col for col in desired_cols if col in df_aba.columns]
    df_final = df_aba[cols_to_keep]
    
    # บันทึกไฟล์
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'amr-a_baumannii_selected.csv')
    df_final.to_csv(save_path, index=False)
    print(f"✅ Step 1 Complete: Saved selected data to {save_path}")

if __name__ == "__main__":
    step1_extract()