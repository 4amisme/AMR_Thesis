import pandas as pd
import os

# 1. กำหนดเส้นทางไฟล์
input_path = os.path.join("MDR", "DrugClass2", "AllYears_MDR_Pattern.csv")
base_output_dir = os.path.join("MDR", "model")

os.makedirs(base_output_dir, exist_ok=True)

try:
    # 2. โหลดข้อมูล (เพิ่ม low_memory=False เพื่อแก้ Warning)
    df = pd.read_csv(input_path, low_memory=False)

    # 3. Data Cleaning & Standardization
    # ตัดแถวที่ข้อมูลสำคัญเป็นค่าว่าง
    df = df.dropna(subset=['ward_type', 'spec_date', 'organism_full', 'Resistant_Drug_Classes'])
    
    # ปรับ ward_type: ตัวเล็ก, ตัดช่องว่าง, ยุบรวมกลุ่ม ICU
    df['ward_type'] = df['ward_type'].astype(str).str.strip().str.lower()
    df['ward_type'] = df['ward_type'].replace({'ccu': 'icu'})
    
    # กรองเฉพาะ ward_type ที่ต้องการ
    target_wards = ['icu', 'in', 'out']
    df = df[df['ward_type'].isin(target_wards)].copy()

    # จัดการเรื่องวันที่
    df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
    df = df.dropna(subset=['spec_date'])
    df = df[(df['spec_date'].dt.year >= 2015) & (df['spec_date'].dt.year <= 2024)].copy()
    
    df['year'] = df['spec_date'].dt.year.astype(int)
    df['month'] = df['spec_date'].dt.month.astype(int)

    # 4. วนลูปประมวลผลรายเชื้อ
    all_organisms = df['organism_full'].unique()
    print(f"กำลังประมวลผลเชื้อทั้งหมด {len(all_organisms)} ชนิด...")

    for organism in all_organisms:
        # กรองข้อมูลเชื้อปัจจุบัน
        df_org = df[df['organism_full'] == organism].copy()
        
        # หา Top 5 Patterns
        top_5_patterns = df_org['Resistant_Drug_Classes'].value_counts().nlargest(5).index.tolist()
        if not top_5_patterns: continue

        # คำนวณตัวหาร (ยอดรวมราย Ward รายเดือน)
        monthly_ward_total = df_org.groupby(['year', 'month', 'ward_type']).size().reset_index(name='total_rows_in_ward_month')

        # คำนวณตัวตั้ง (เฉพาะ Top 5)
        df_top5 = df_org[df_org['Resistant_Drug_Classes'].isin(top_5_patterns)]
        monthly_ward_counts = df_top5.groupby(['year', 'month', 'ward_type', 'Resistant_Drug_Classes']).size().reset_index(name='pattern_count')

        # รวมข้อมูล
        final_df_org = pd.merge(monthly_ward_counts, monthly_ward_total, on=['year', 'month', 'ward_type'])
        final_df_org['percentage'] = ((final_df_org['pattern_count'] / final_df_org['total_rows_in_ward_month']) * 100).round(2)
        
        # ใส่ชื่อเชื้อกลับเข้าไปใน DataFrame
        final_df_org['organism_full'] = organism

        # สร้าง Folder เชื้อ
        clean_org_folder = "".join([c if c.isalnum() else "_" for c in str(organism).lower()])
        org_dir = os.path.join(base_output_dir, clean_org_folder)
        os.makedirs(org_dir, exist_ok=True)

        # ย่อชื่อเชื้อสำหรับชื่อไฟล์
        parts = str(organism).lower().split()
        short_org_name = f"{parts[0][0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
        short_org_name = "".join([c if c.isalnum() or c == '_' else "" for c in short_org_name])

        # 5. บันทึกแยกไฟล์ราย Ward
        for ward in final_df_org['ward_type'].unique():
            ward_df = final_df_org[final_df_org['ward_type'] == ward].copy()
            
            # จัดลำดับคอลัมน์ (ตรวจสอบให้แน่ใจว่า organism_full อยู่ใน list)
            column_order = [
                'organism_full', 'year', 'month', 'ward_type', 
                'Resistant_Drug_Classes', 'pattern_count', 
                'total_rows_in_ward_month', 'percentage'
            ]
            
            # เลือกเฉพาะคอลัมน์ที่มีอยู่จริงป้องกัน error
            final_columns = [col for col in column_order if col in ward_df.columns]
            ward_df = ward_df[final_columns].sort_values(['year', 'month', 'percentage'], ascending=[True, True, False])

            file_name = f"{short_org_name}_{ward}.csv"
            ward_df.to_csv(os.path.join(org_dir, file_name), index=False)

    print(f"--- ดำเนินการเสร็จสมบูรณ์ ---")

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")