import pandas as pd
import os
import numpy as np

INPUT_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/data/MDR/AllYears_DrugClass_tested.csv'
OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR'

def step1_extract_loop():
    # 1. โหลดข้อมูลครั้งเดียวข้างนอก Loop เพื่อประหยัดเวลาและ Memory
    print("⏳ Loading master data...")
    df_master = pd.read_csv(INPUT_PATH, low_memory=False)
    df_master.columns = df_master.columns.str.lower()
    
    # กรองเฉพาะปีที่สนใจก่อน
    df_master = df_master[(df_master['x_year'] >= 2015) & (df_master['x_year'] <= 2024)].copy()
    
    # 2. กำหนดรายชื่อเชื้อที่ต้องการรัน (รวม E. faecalis และที่เหลือ)
    target_organisms = [
        'Enterococcus faecalis',
        'Staphylococcus aureus',
        'Enterococcus faecium',
        'Escherichia coli',
        'Klebsiella pneumoniae ',
        'Pseudomonas aeruginosa '
    ]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. เริ่ม Loop รันทีละเชื้อ
    for org_name in target_organisms:
        print(f"\n🔍 Processing: {org_name}...")
        
        # กรองเฉพาะเชื้อปัจจุบัน
        df_sub = df_master[df_master['organism_full'].str.contains(org_name, case=False, na=False)].copy()
        
        if df_sub.empty:
            print(f"⚠️ No data found for {org_name}, skipping.")
            continue

        # จัดการ ward_type
        df_sub['ward_type'] = df_sub['ward_type'].astype(str).str.lower().str.strip()
        df_sub['ward_type'] = df_sub['ward_type'].replace(['nan', 'none', '', 'null'], np.nan)
        
        # จัดการ x_spec_ful
        df_sub['x_spec_ful'] = df_sub['x_spec_ful'].astype(str).str.strip()
        df_sub['x_spec_ful'] = df_sub['x_spec_ful'].replace(['nan', 'none', '', 'null'], np.nan)
        
        # หา Top 3 Specimens ของเชื้อนั้นๆ
        valid_specs = df_sub.dropna(subset=['x_spec_ful'])
        top3 = valid_specs['x_spec_ful'].value_counts().nlargest(3).index.tolist()
        print(f"📌 {org_name} - Top 3 Specimens: {top3}")
        
        # สร้าง spec_group
        df_sub['spec_group'] = df_sub['x_spec_ful'].apply(
            lambda x: x if x in top3 else ('other' if pd.notnull(x) else np.nan)
        )
        
        # เลือกคอลัมน์และบันทึกไฟล์ (แยกชื่อไฟล์ตามชื่อเชื้อ)
        cols_to_keep = ['organism_full', 'x_year', 'region', 'ward_type', 'x_spec_ful', 'spec_group']
        df_final = df_sub[cols_to_keep]
        
        # สร้างชื่อไฟล์ให้ไม่มีเว้นวรรค (เช่น e_coli_selected.csv)
        file_name = org_name.lower().replace(' ', '_').replace('.', '') + '_selected.csv'
        output_file = os.path.join(OUTPUT_DIR, file_name)
        
        df_final.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")

if __name__ == "__main__":
    step1_extract_loop()
    print("\n🚀 All organisms processed successfully!")