import pandas as pd
import os

# 1. กำหนดเส้นทางไฟล์
input_path = os.path.join("MDR", "DrugClass2", "AllYears_MDR_Pattern.csv")
base_output_dir = os.path.join("MDR", "model_by_specimen")

os.makedirs(base_output_dir, exist_ok=True)

try:
    # 2. โหลดข้อมูล (low_memory=False เพื่อจัดการ mixed types)
    df = pd.read_csv(input_path, low_memory=False)

    # 3. Data Cleaning
    # ตัด row ที่สำคัญที่เป็น null
    df = df.dropna(subset=['spec_type', 'spec_date', 'organism_full', 'Resistant_Drug_Classes'])
    
    # ทำความสะอาด spec_type (ตัวเล็ก, ตัดช่องว่าง)
    df['spec_type'] = df['spec_type'].astype(str).str.strip().str.lower()

    # จัดการเรื่องวันที่ (2015 - 2024)
    df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
    df = df.dropna(subset=['spec_date'])
    df = df[(df['spec_date'].dt.year >= 2015) & (df['spec_date'].dt.year <= 2024)].copy()
    
    df['year'] = df['spec_date'].dt.year.astype(int)
    df['month'] = df['spec_date'].dt.month.astype(int)

    # 4. วนลูปประมวลผลรายเชื้อ
    all_organisms = df['organism_full'].unique()
    print(f"กำลังประมวลผลเชื้อทั้งหมด {len(all_organisms)} ชนิด...")

    for organism in all_organisms:
        df_org = df[df['organism_full'] == organism].copy()
        
        # ก. หา Global Top 5 Resistant_Drug_Classes ของเชื้อนี้
        top_5_patterns = df_org['Resistant_Drug_Classes'].value_counts().nlargest(5).index.tolist()
        
        # ข. หา Top 5 spec_type ของเชื้อนี้
        top_5_specs = df_org['spec_type'].value_counts().nlargest(5).index.tolist()
        
        if not top_5_patterns or not top_5_specs: continue

        # กรองข้อมูลเอาเฉพาะ Top 5 spec_type เพื่อความรวดเร็ว
        df_top_specs = df_org[df_org['spec_type'].isin(top_5_specs)].copy()

        # ค. คำนวณตัวหาร: จำนวนเคสรวมแยกตาม [Year, Month, spec_type]
        monthly_spec_total = df_top_specs.groupby(['year', 'month', 'spec_type']).size().reset_index(name='total_rows_in_spec_month')

        # ง. คำนวณตัวตั้ง: เฉพาะ Top 5 Drug Patterns ใน Top 5 spec_type
        df_final_targets = df_top_specs[df_top_specs['Resistant_Drug_Classes'].isin(top_5_patterns)]
        monthly_counts = df_final_targets.groupby(['year', 'month', 'spec_type', 'Resistant_Drug_Classes']).size().reset_index(name='pattern_count')

        # จ. รวมข้อมูลและคำนวณ %
        final_df_org = pd.merge(monthly_counts, monthly_spec_total, on=['year', 'month', 'spec_type'])
        final_df_org['percentage'] = ((final_df_org['pattern_count'] / final_df_org['total_rows_in_spec_month']) * 100).round(2)
        final_df_org['organism_full'] = organism

        # 5. บันทึกไฟล์แยกตาม spec_type
        # สร้าง Folder เชื้อ
        clean_org_folder = "".join([c if c.isalnum() else "_" for c in str(organism).lower()])
        org_dir = os.path.join(base_output_dir, clean_org_folder)
        os.makedirs(org_dir, exist_ok=True)

        # ย่อชื่อเชื้อสำหรับชื่อไฟล์
        parts = str(organism).lower().split()
        short_org_name = f"{parts[0][0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
        short_org_name = "".join([c if c.isalnum() or c == '_' else "" for c in short_org_name])

        for spec in final_df_org['spec_type'].unique():
            spec_df = final_df_org[final_df_org['spec_type'] == spec].copy()
            
            column_order = [
                'organism_full', 'year', 'month', 'spec_type', 
                'Resistant_Drug_Classes', 'pattern_count', 
                'total_rows_in_spec_month', 'percentage'
            ]
            
            # เรียงตามเวลาและ %
            spec_df = spec_df[column_order].sort_values(['year', 'month', 'percentage'], ascending=[True, True, False])

            # ชื่อไฟล์: a_baumannii_sputum.csv
            clean_spec_name = "".join([c if c.isalnum() else "_" for c in spec])
            file_name = f"{short_org_name}_{clean_spec_name}.csv"
            spec_df.to_csv(os.path.join(org_dir, file_name), index=False)

    print(f"--- ดำเนินการเสร็จสมบูรณ์ ---")
    print(f"ตรวจสอบไฟล์ที่แยกตาม specimen ได้ที่: {base_output_dir}")

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")