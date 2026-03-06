import pandas as pd
import numpy as np
import os

# แก้ไข Path ให้ตรงกับเครื่องของคุณ
INPUT_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend/amr-a_baumannii_selected.csv' 
BASE_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend/All data'

def create_satscan_case_files():
    print("⏳ กำลังเตรียมไฟล์ Case Data สำหรับแต่ละยา...")
    
    # โหลดข้อมูล
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    if 'spec_date' not in df.columns:
        print("❌ ไม่พบคอลัมน์ 'spec_date'")
        return
        
    df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
    df = df.dropna(subset=['spec_date', 'region'])
    
    # กรองเอาเฉพาะปี 2015 - 2024
    df = df[(df['spec_date'].dt.year >= 2015) & (df['spec_date'].dt.year <= 2024)].copy()
    
    # สร้างคอลัมน์ตามที่ระบุ
    df['year'] = df['spec_date'].dt.year
    df['date'] = df['spec_date'].dt.strftime('%Y/%m') 
    df['region'] = df['region'].astype(int)

    def process_drug(data, drug_col, out_filename):
        if drug_col not in data.columns:
            print(f"⚠️ ไม่พบคอลัมน์ {drug_col}")
            return
            
        df_drug = data.dropna(subset=[drug_col]).copy()
        # เช็ค R = 1, อื่นๆ = 0
        df_drug['is_resistant'] = df_drug[drug_col].astype(str).str.upper().apply(lambda x: 1 if x == 'R' else 0)
        
        # 🌟 จุดที่มีการแก้ไข: จัดกลุ่มตามคอลัมน์ใหม่ (รวมรายเดือน ตัด ward_type และ spec_group ออก)
        group_cols = ['region', 'organism_full', 'year', 'date']
        
        agg_df = df_drug.groupby(group_cols).agg(
            n_tested=('is_resistant', 'count'),
            n_resistant=('is_resistant', 'sum')
        ).reset_index()
        
        # เพิ่มคอลัมน์ชื่อยาและคำนวณ percent_R
        agg_df['drug'] = drug_col.capitalize()
        agg_df['percent_R'] = ((agg_df['n_resistant'] / agg_df['n_tested']) * 100).round(2)
        
        # เปลี่ยนชื่อคอลัมน์ให้ตรงกับ final_cols
        agg_df = agg_df.rename(columns={'organism_full': 'organism'})
        
        # จัดเรียงลำดับคอลัมน์ตามที่กำหนดเป๊ะๆ
        final_cols = ['region', 'organism', 'year', 'date', 'n_tested', 'n_resistant', 'drug', 'percent_R']
        agg_df = agg_df[final_cols]
        
        # เซฟเป็น CSV
        out_path = os.path.join(BASE_DIR, out_filename)
        agg_df.to_csv(out_path, index=False)
        print(f"✅ บันทึกไฟล์สำเร็จ: {out_filename}")

    # รันทั้งสองตัวยา
    process_drug(df, 'imipenem', 'SaTScan_Case_Imipenem.csv')
    process_drug(df, 'meropenem', 'SaTScan_Case_Meropenem.csv')

if __name__ == "__main__":
    create_satscan_case_files()