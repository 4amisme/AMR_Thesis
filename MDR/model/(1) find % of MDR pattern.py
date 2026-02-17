import pandas as pd
import os

# 1. กำหนดเส้นทางไฟล์
input_path = os.path.join("MDR", "DrugClass2", "AllYears_MDR_Pattern.csv")
output_dir = os.path.join("MDR", "model")

os.makedirs(output_dir, exist_ok=True)

try:
    # 2. โหลดข้อมูล
    df = pd.read_csv(input_path)

    # 3. จัดการเรื่องวันที่ (ป้องกัน Error จากปีที่อยู่นอกขอบเขต)
    df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
    df = df.dropna(subset=['spec_date'])

    # กรองช่วงปี 2015 - 2024
    df = df[(df['spec_date'].dt.year >= 2015) & (df['spec_date'].dt.year <= 2024)].copy()

    # สร้างคอลัมน์ year และ month
    df['year'] = df['spec_date'].dt.year.astype(int)
    df['month'] = df['spec_date'].dt.month.astype(int)

    # 4. วนลูปประมวลผลรายเชื้อ (organism_full)
    all_organisms = df['organism_full'].unique()
    print(f"กำลังประมวลผลเชื้อทั้งหมด {len(all_organisms)} ชนิด...")

    for organism in all_organisms:
        if pd.isna(organism): continue
        
        df_org = df[df['organism_full'] == organism].copy()
        
        # หา Global Top 5 Resistant_Drug_Classes สำหรับเชื้อนี้
        top_5_patterns = df_org['Resistant_Drug_Classes'].value_counts().nlargest(5).index.tolist()
        
        if not top_5_patterns: continue

        # คำนวณจำนวนเคสรวมรายเดือน (ตัวหาร)
        monthly_total = df_org.groupby(['year', 'month']).size().reset_index(name='total_rows_in_month')

        # คำนวณจำนวนการเกิดของแต่ละ Pattern ใน Top 5
        df_top5 = df_org[df_org['Resistant_Drug_Classes'].isin(top_5_patterns)]
        monthly_counts = df_top5.groupby(['year', 'month', 'Resistant_Drug_Classes']).size().reset_index(name='pattern_count')

        # รวมข้อมูลและคำนวณ %
        final_df = pd.merge(monthly_counts, monthly_total, on=['year', 'month'])
        final_df['percentage'] = ((final_df['pattern_count'] / final_df['total_rows_in_month']) * 100).round(2)
        
        # เพิ่มคอลัมน์ organism_full
        final_df['organism_full'] = organism

        # --- จัดลำดับคอลัมน์ใหม่ (ตัดตัวที่ซ้ำออกแล้ว) ---
        column_order = [
            'organism_full', 
            'year', 
            'month', 
            'Resistant_Drug_Classes', 
            'pattern_count', 
            'total_rows_in_month', 
            'percentage'
        ]
        
        final_df = final_df[column_order]
        
        # เรียงลำดับตามเวลาและเปอร์เซ็นต์
        final_df = final_df.sort_values(['year', 'month', 'percentage'], ascending=[True, True, False])

        # 5. บันทึกไฟล์ (ล้างชื่อไฟล์ให้ปลอดภัย)
        clean_name = "".join([c if c.isalnum() else "_" for c in str(organism).lower()])
        file_name = f"{clean_name}.csv"
        final_df.to_csv(os.path.join(output_dir, file_name), index=False)

    print(f"--- ดำเนินการเสร็จสมบูรณ์ ---")
    print(f"ตรวจสอบไฟล์ทั้งหมดได้ที่: {output_dir}")

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")