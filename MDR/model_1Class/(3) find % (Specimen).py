import pandas as pd
import os

# 1. กำหนดเส้นทางไฟล์
input_path = os.path.join("MDR", "DrugClass2", "AllYears_SingleClass_Pattern.csv")
base_output_dir = os.path.join("MDR", "model_1Class")

os.makedirs(base_output_dir, exist_ok=True)

try:
    # 2. โหลดข้อมูล
    df = pd.read_csv(input_path, low_memory=False)
    
    # ปรับหัวตารางเป็นตัวเล็ก
    df.columns = df.columns.str.strip().str.lower()
    target_pattern_col = 'resistant_drug_classes' 

    # --- ส่วนแก้ไขเรื่อง ICU, icu, Icu ---
    # 3. จัดการข้อมูลเบื้องต้น (Standardize Ward Type)
    # ตัดแถวที่เป็นค่าว่างในคอลัมน์สำคัญออกก่อน
    df = df.dropna(subset=['ward_type', 'spec_date', 'organism_full', target_pattern_col])
    
    # แปลงเป็น String -> ตัดช่องว่าง -> แปลงเป็นตัวพิมพ์เล็กทั้งหมด (แก้ปัญหา ICU/icu/Icu)
    df['ward_type'] = df['ward_type'].astype(str).str.strip().str.lower()
    
    # ยุบรวม CCU เป็น ICU (เนื่องจากเป็นตัวเล็กหมดแล้ว จึงใช้ 'ccu' และ 'icu')
    df['ward_type'] = df['ward_type'].replace({'ccu': 'icu'})
    
    # กรองเฉพาะ ward_type ที่ต้องการ
    target_wards = ['icu', 'in', 'out']
    df = df[df['ward_type'].isin(target_wards)].copy()

    # 4. จัดการเรื่องวันที่
    df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
    df = df.dropna(subset=['spec_date'])
    df = df[(df['spec_date'].dt.year >= 2015) & (df['spec_date'].dt.year <= 2024)].copy()
    
    df['year'] = df['spec_date'].dt.year.astype(int)
    df['month'] = df['spec_date'].dt.month.astype(int)

    # 5. วนลูปประมวลผลรายเชื้อ
    all_organisms = df['organism_full'].unique()
    print(f"กำลังประมวลผลเชื้อทั้งหมด {len(all_organisms)} ชนิด (แบบครบทุก Pattern)...")

    for organism in all_organisms:
        if pd.isna(organism): continue
        df_org = df[df['organism_full'] == organism].copy()
        
        # คำนวณตัวหาร: ยอดรวมของ Ward นั้นในเดือนนั้น
        monthly_ward_total = df_org.groupby(['year', 'month', 'ward_type']).size().reset_index(name='total_rows_in_ward_month')

        # คำนวณตัวตั้ง: นับทุก Pattern ที่เกิดขึ้น
        monthly_ward_counts = df_org.groupby(['year', 'month', 'ward_type', target_pattern_col]).size().reset_index(name='pattern_count')

        # รวมข้อมูล
        final_df_org = pd.merge(monthly_ward_counts, monthly_ward_total, on=['year', 'month', 'ward_type'])
        final_df_org['percentage'] = ((final_df_org['pattern_count'] / final_df_org['total_rows_in_ward_month']) * 100).round(2)
        final_df_org['organism_full'] = organism

        # 6. สร้าง Folder และบันทึกไฟล์
        clean_org_folder = "".join([c if c.isalnum() else "_" for c in str(organism).lower()])
        org_dir = os.path.join(base_output_dir, clean_org_folder)
        os.makedirs(org_dir, exist_ok=True)

        parts = str(organism).lower().split()
        short_org_name = f"{parts[0][0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
        short_org_name = "".join([c if c.isalnum() or c == '_' else "" for c in short_org_name])

        for ward in final_df_org['ward_type'].unique():
            ward_df = final_df_org[final_df_org['ward_type'] == ward].copy()
            column_order = ['organism_full', 'year', 'month', 'ward_type', target_pattern_col, 'pattern_count', 'total_rows_in_ward_month', 'percentage']
            final_columns = [col for col in column_order if col in ward_df.columns]
            ward_df = ward_df[final_columns].sort_values(['year', 'month', 'percentage'], ascending=[True, True, False])

            file_name = f"{short_org_name}_{ward}.csv"
            ward_df.to_csv(os.path.join(org_dir, file_name), index=False)

    print(f"--- ดำเนินการเสร็จสมบูรณ์ (ข้อมูลครบถ้วนและ Ward เป็นตัวเล็กทั้งหมด) ---")

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")