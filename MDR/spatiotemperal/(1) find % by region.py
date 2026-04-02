import pandas as pd
import os

# 1. กำหนดเส้นทางไฟล์
input_path = os.path.join("MDR", "DrugClass2", "AllYears_MDR_Pattern.csv")
base_output_dir = os.path.join("MDR", "spatiotemperal")

os.makedirs(base_output_dir, exist_ok=True)

try:
    # 2. โหลดข้อมูล
    df = pd.read_csv(input_path, low_memory=False)

    # 3. Data Cleaning
    df = df.dropna(subset=['region', 'spec_date', 'organism_full', 'Resistant_Drug_Classes'])
    df['region'] = df['region'].astype(str).str.strip()

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
        
        # ก. หา Global Top 5 Resistant_Drug_Classes สำหรับเชื้อนี้
        top_5_patterns = df_org['Resistant_Drug_Classes'].value_counts().nlargest(5).index.tolist()
        if not top_5_patterns: continue

        # ข. คำนวณตัวหาร: จำนวนเคสรวมแยกตาม [Year, Month, Region]
        monthly_region_total = df_org.groupby(['year', 'month', 'region']).size().reset_index(name='total_rows_in_region_month')

        # ค. คำนวณตัวตั้ง: เฉพาะ Top 5 Patterns แยกตาม [Year, Month, Region]
        df_top5 = df_org[df_org['Resistant_Drug_Classes'].isin(top_5_patterns)]
        monthly_counts = df_top5.groupby(['year', 'month', 'region', 'Resistant_Drug_Classes']).size().reset_index(name='pattern_count')

        # ง. รวมข้อมูลและคำนวณ %
        final_df_org = pd.merge(monthly_counts, monthly_region_total, on=['year', 'month', 'region'])
        final_df_org['percentage'] = ((final_df_org['pattern_count'] / final_df_org['total_rows_in_region_month']) * 100).round(2)
        final_df_org['organism_full'] = organism

        # 5. การจัดการโฟลเดอร์และชื่อไฟล์
        clean_org_folder = "".join([c if c.isalnum() else "_" for c in str(organism).lower()])
        org_dir = os.path.join(base_output_dir, clean_org_folder)
        os.makedirs(org_dir, exist_ok=True)

        parts = str(organism).lower().split()
        short_org_name = f"{parts[0][0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
        short_org_name = "".join([c if c.isalnum() or c == '_' else "" for c in short_org_name])

        # จัดลำดับคอลัมน์มาตรฐาน
        column_order = [
            'organism_full', 'year', 'month', 'region', 
            'Resistant_Drug_Classes', 'pattern_count', 
            'total_rows_in_region_month', 'percentage'
        ]

        # --- ส่วนที่ 1: บันทึกไฟล์แยกแต่ละ Region (เหมือนเดิม) ---
        for region_val in final_df_org['region'].unique():
            region_df = final_df_org[final_df_org['region'] == region_val].copy()
            region_df = region_df[column_order].sort_values(['year', 'month', 'percentage'], ascending=[True, True, False])
            
            clean_region_name = region_val.lower().replace(" ", "_")
            file_name = f"{short_org_name}_{clean_region_name}.csv"
            region_df.to_csv(os.path.join(org_dir, file_name), index=False)

        # --- ส่วนที่ 2: บันทึกไฟล์รวมทุกเขตสุขภาพ (Summary File) ---
        summary_file_name = f"{short_org_name}_all_regions.csv"
        # เรียงตาม ปี > เดือน > เขต > % เพื่อให้ดูง่ายเวลาเปรียบเทียบข้ามเขต
        summary_df = final_df_org[column_order].sort_values(['year', 'month', 'region', 'percentage'], 
                                                            ascending=[True, True, True, False])
        summary_df.to_csv(os.path.join(org_dir, summary_file_name), index=False)

    print(f"--- ดำเนินการเสร็จสมบูรณ์ ---")
    print(f"บันทึกไฟล์แยก Region และไฟล์รวมสรุปไว้ที่: {base_output_dir}")

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")