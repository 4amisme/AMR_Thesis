import pandas as pd
import os

# ==============================================================================
# 1. การตั้งค่า (CONFIG)
# ==============================================================================
# โฟลเดอร์ที่เก็บไฟล์
BASE_FOLDER = os.path.join("MDR", "DrugClass2")

# รายชื่อไฟล์เชื้อทั้งหมด 6 ไฟล์
TARGET_FILES = [
    "acinetobacter_baumannii.csv",
    "enterococcus_faecalis.csv",
    "enterococcus_faecium.csv",
    "escherichia_coli.csv",
    "klebsiella_pneumoniae.csv",
    "pseudomonas_aeruginosa.csv"
]

# ไฟล์มาตรฐาน (Mapping และ ID)
FILE_MAPPING = os.path.join(BASE_FOLDER, "Drug_class_for_MDR_new.csv")
FILE_ID = os.path.join(BASE_FOLDER, "ID.csv")

# ==============================================================================
# 2. ฟังก์ชันช่วยงาน (HELPER FUNCTIONS)
# ==============================================================================

def get_short_name(filename):
    """
    แปลงชื่อไฟล์ยาวๆ ให้เป็นชื่อสั้นๆ สำหรับตั้งชื่อไฟล์ Output
    Ex: 'acinetobacter_baumannii.csv' -> 'A_baumannii'
    """
    name = filename.replace('.csv', '')
    parts = name.split('_')
    if len(parts) >= 2:
        # เอาตัวอักษรแรกของคำหน้า + คำหลังเต็มๆ (เช่น A_baumannii)
        short_name = f"{parts[0][0].upper()}_{parts[1]}"
        return short_name
    return name

def normalize_string(text):
    """
    จัดรูปแบบข้อความรายการยา: แยก comma -> ตัดช่องว่าง -> เรียง A-Z -> รวมกลับ
    """
    if pd.isna(text) or str(text).strip() == "" or str(text).lower() == 'nan':
        return ""
    parts = [p.strip() for p in str(text).split(',')]
    parts.sort()
    return ", ".join(parts)

# ==============================================================================
# 3. MAIN LOGIC
# ==============================================================================

def main():
    print(f"--- เริ่มต้นกระบวนการประมวลผล {len(TARGET_FILES)} ไฟล์เชื้อ ---")
    
    # --- STEP 3.1: โหลดไฟล์มาตรฐาน (Mapping & ID) ---
    if not os.path.exists(FILE_MAPPING) or not os.path.exists(FILE_ID):
        print(f"[Critical Error] ไม่พบไฟล์ Mapping หรือ ID ใน {BASE_FOLDER}")
        return

    print(">> กำลังโหลดไฟล์มาตรฐาน (Mapping & ID)...")
    try:
        df_map = pd.read_csv(FILE_MAPPING, encoding='utf-8')
        df_id = pd.read_csv(FILE_ID, encoding='utf-8')
    except Exception as e:
        print(f"Error loading standard files: {e}")
        return

    # 3.1.1 เตรียมข้อมูล Mapping
    # Clean ชื่อคอลัมน์
    df_map.columns = df_map.columns.str.strip()
    
    # สร้าง Dictionary: (ชื่อเชื้อ, ชื่อยา) -> Class
    organism_drug_map = {}
    for _, row in df_map.iterrows():
        # แปลงเป็นตัวพิมพ์เล็กทั้งหมดเพื่อป้องกันปัญหา Case Sensitive
        org = str(row['ORGANISM_WHO']).strip().lower()
        drug = str(row['Antibiotic']).strip().lower()
        cls = str(row['Class']).strip()
        organism_drug_map[(org, drug)] = cls

    # 3.1.2 เตรียมข้อมูล ID
    df_id.columns = df_id.columns.str.strip()
    
    # แก้ไขตัวอักษร Encoding ผิด (Î² -> β) ในคอลัมน์ std_tested_list
    if 'std_tested_list' in df_id.columns:
        print(">> แก้ไข Encoding (Î² -> β) ในไฟล์ ID...")
        df_id['std_tested_list'] = df_id['std_tested_list'].astype(str).str.replace("Î²-lactam", "β-lactam", regex=False)
    
    # สร้าง Key สำหรับ Join (Sort ยา A-Z)
    df_id['join_key'] = df_id['std_tested_list'].apply(normalize_string)

    print(">> โหลดไฟล์มาตรฐานเรียบร้อย\n")

    # --- STEP 3.2: วนลูปทำทีละไฟล์เชื้อ ---
    for filename in TARGET_FILES:
        input_path = os.path.join(BASE_FOLDER, filename)
        
        # ตั้งชื่อไฟล์ Output แบบย่อ (เช่น MDR_A_baumannii.csv)
        short_name = get_short_name(filename)
        output_filename = f"MDR_{short_name}.csv"
        output_path = os.path.join(BASE_FOLDER, output_filename)
        
        print(f"Processing: {filename} ...")
        
        if not os.path.exists(input_path):
            print(f"   [Skip] ไม่พบไฟล์ {filename}")
            continue

        try:
            # 1. อ่านไฟล์เชื้อ
            df = pd.read_csv(input_path, encoding='utf-8')
            df.columns = df.columns.str.strip()

            # ตรวจสอบคอลัมน์สำคัญ
            if 'organism_full' not in df.columns or 'x_year' not in df.columns:
                print(f"   [Error] ไฟล์ {filename} ขาดคอลัมน์ 'organism_full' หรือ 'x_year'")
                continue
            
            # ลบช่องว่างในชื่อเชื้อ
            df['organism_full'] = df['organism_full'].astype(str).str.strip()

            # 2. หา Resistance Class (Row by Row)
            #    ฟังก์ชันนี้จะทำงานภายใน Loop เพื่อใช้ตัวแปร df ปัจจุบัน
            def get_resistance(row):
                found = set()
                # ดึงชื่อเชื้อจากแถวปัจจุบัน (แปลงเป็นตัวเล็กเพื่อเทียบกับ Mapping)
                curr_org = str(row['organism_full']).lower()
                
                for col in df.columns:
                    # ข้ามคอลัมน์ที่ไม่ใช่ยา
                    if col.lower() in ['sample_id', 'organism_full', 'x_year']: continue
                    
                    # สร้าง Key เพื่อตรวจสอบ Mapping
                    key = (curr_org, col.strip().lower())
                    if key in organism_drug_map:
                        # ถ้าผลเป็น R (รองรับตัวเล็ก/ใหญ่)
                        if str(row[col]).strip().lower() == 'r':
                            found.add(organism_drug_map[key])
                
                if not found: return ""
                # คืนค่ากลับโดยเรียง A-Z
                return ", ".join(sorted(list(found)))

            # Apply ฟังก์ชัน
            df['Resistant_Drug_Classes'] = df.apply(get_resistance, axis=1)

            # 3. กรอง MDR (>= 3 Classes)
            # นับจำนวน comma + 1 (หรือ 0 ถ้าว่าง)
            df['Class_Count'] = df['Resistant_Drug_Classes'].apply(lambda x: len(x.split(',')) if x else 0)
            
            # *** Filter MDR ***
            mdr_df = df[df['Class_Count'] >= 3].copy()

            if mdr_df.empty:
                print(f"   [Info] {filename} ไม่พบเคส MDR (>=3 classes) เลย -> ข้าม")
                continue

            # 4. สรุปยอด (Group By)
            # Group โดย: ปี + ชื่อเชื้อ + รูปแบบการดื้อยา
            summary = mdr_df.groupby(['x_year', 'organism_full', 'Resistant_Drug_Classes']).size().reset_index(name='Count')
            
            # หาผลรวมแยกตามปีและเชื้อ (Total MDR per Year per Organism)
            total_mdr = mdr_df.groupby(['x_year', 'organism_full']).size().reset_index(name='Total_MDR')
            
            # Merge เพื่อคำนวณ %
            summary = pd.merge(summary, total_mdr, on=['x_year', 'organism_full'])
            summary['Percentage'] = (summary['Count'] / summary['Total_MDR'] * 100).round(2)
            
            # เรียงลำดับข้อมูล
            summary = summary.sort_values(['x_year', 'Count'], ascending=[True, False])

            # 5. จับคู่กับ ID (Mapping ID)
            summary['join_key'] = summary['Resistant_Drug_Classes'].apply(normalize_string)
            
            # Merge กับตาราง ID
            final_df = pd.merge(summary, df_id[['join_key', 'MDR_id']], on='join_key', how='left')
            final_df.drop(columns=['join_key'], inplace=True)
            
            # จัดลำดับคอลัมน์
            cols_order = ['x_year', 'organism_full', 'Resistant_Drug_Classes', 'Count', 'Total_MDR', 'Percentage', 'MDR_id']
            final_cols = [c for c in cols_order if c in final_df.columns]
            final_df = final_df[final_cols]

            # 6. บันทึกไฟล์ (กลับมาใช้ utf-8 ปกติ)
            final_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"   ✅ Saved: {output_filename}")

        except Exception as e:
            print(f"   [Error] เกิดข้อผิดพลาดกับไฟล์ {filename}: {e}")

    print("\n--- เสร็จสิ้นทุกขั้นตอน ---")

if __name__ == "__main__":
    main()