import pandas as pd
import os

def main():
    # --- 1. CONFIG PATH ---
    mapping_folder = os.path.join("MDR", "DrugClass2")
    input_file_mapping = os.path.join(mapping_folder, "Drug_class_for_MDR_new.csv")

    input_folder = os.path.join("MDR", "DrugClass2")
    input_file_main = os.path.join(input_folder, "AllYears_DrugClass_tested.csv")
    
    # เปลี่ยนชื่อไฟล์ output ให้สะท้อนว่าเป็น Single Class Resistance
    output_path = os.path.join(mapping_folder, "AllYears_SingleClass_Pattern.csv")

    print("--- เริ่มต้นกระบวนการ (คัดกรองดื้อยา 1 กลุ่มเท่านั้น) ---")

    # --- 2. LOAD FILES ---
    try:
        df_main = pd.read_csv(input_file_main, encoding='utf-8')
        df_mapping = pd.read_csv(input_file_mapping, encoding='utf-8')
        print(f"✅ โหลดไฟล์สำเร็จ")
    except Exception as e:
        print(f"[Error] {e}")
        return

    # --- 3. CLEANING ---
    # แปลงชื่อคอลัมน์เป็นตัวเล็กและตัดช่องว่าง
    df_main.columns = df_main.columns.str.strip().str.lower()
    df_mapping.columns = df_mapping.columns.str.strip().str.lower()
    
    # ตรวจสอบคอลัมน์สำคัญ
    if 'organism_full' not in df_main.columns:
        print("❌ ไม่พบคอลัมน์ 'organism_full'")
        return

    # --- 4. FILTER Missing_Count = 0 ---
    target_col = 'missing_count'
    if target_col in df_main.columns:
        print(f"\n[Filter] กรอง {target_col} == 0 ...")
        df_main = df_main[df_main[target_col] == 0].copy()
        print(f"   - เหลือข้อมูลสำหรับวิเคราะห์: {len(df_main)} แถว")

    # --- 5. MAPPING PREP ---
    print("\n[Processing] กำลังสร้างแผนผังเชื้อและกลุ่มยา...")
    
    organism_drug_map = {}
    for _, row in df_mapping.iterrows():
        org = str(row['organism_who']).strip().lower()
        drug = str(row['antibiotic']).strip().lower()
        cls = str(row['class']).strip()
        organism_drug_map[(org, drug)] = cls

    # --- 6. FIND RESISTANCE COUNT ---
    # รายชื่อยาคือคอลัมน์ที่ไม่ใช่ข้อมูลพื้นฐาน
    exclude_cols = ['sample_id', 'organism_full', 'x_year', 'missing_count', 'resistant_drug_classes', 'class_count']
    drug_cols = [c for c in df_main.columns if c not in exclude_cols]
    
    def get_mdr_classes(row):
        found = set()
        curr_org = str(row['organism_full']).strip().lower()
        
        for col in drug_cols:
            key = (curr_org, col)
            if key in organism_drug_map:
                val = str(row[col]).strip().lower()
                # เช็คผลดื้อยา (R)
                if val in ['r', 'resistant', '1']: 
                    found.add(organism_drug_map[key])
        
        return len(found)

    print("   - กำลังคำนวณจำนวนกลุ่มยาที่ดื้อ...")
    df_main['class_count'] = df_main.apply(get_mdr_classes, axis=1)

    # --- 7. FINAL FILTER (เปลี่ยนเป็นดื้อแค่ 1 กลุ่ม) ---
    # กรองเฉพาะที่ดื้อยา 1 กลุ่มเป๊ะๆ
    single_class_df = df_main[df_main['class_count'] == 1].copy() 
    
    print(f"\n[Result] พบเชื้อที่ดื้อยาเพียง 1 กลุ่ม: {len(single_class_df)} แถว")

    if not single_class_df.empty:
        # ฟังก์ชันดึงชื่อกลุ่มยามาแสดงผล
        def get_class_names(row):
            found = set()
            curr_org = str(row['organism_full']).strip().lower()
            for col in drug_cols:
                key = (curr_org, col)
                if key in organism_drug_map and str(row[col]).strip().lower() in ['r', 'resistant', '1']:
                    found.add(organism_drug_map[key])
            return ", ".join(sorted(list(found)))
            
        single_class_df['resistant_drug_classes'] = single_class_df.apply(get_class_names, axis=1)
        
        # บันทึกไฟล์
        single_class_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ บันทึกเสร็จสมบูรณ์: {output_path}")
        print("-" * 30)
        print(single_class_df[['organism_full', 'class_count', 'resistant_drug_classes']].head(10))
    else:
        print("❌ ไม่พบข้อมูลที่ตรงตามเงื่อนไข (ดื้อยา 1 กลุ่ม)")

if __name__ == "__main__":
    main()