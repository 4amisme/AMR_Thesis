import pandas as pd
import os

# ==============================================================================
# 1. การตั้งค่า (CONFIG)
# ==============================================================================

# 1.1 รายชื่อไฟล์ตั้งต้น (Source Files) - ระบุ Path เต็ม
SOURCE_FILES = [
    r"C:\Users\ucwsh\Downloads\escherichia_coli.csv",
    r"C:\Users\ucwsh\Downloads\klebsiella_pneumoniae.csv",
    r"C:\Users\ucwsh\Downloads\enterococcus_faecium.csv",
    r"C:\Users\ucwsh\Downloads\enterococcus_faecalis.csv",
    r"C:\Users\ucwsh\Downloads\acinetobacter_baumannii.csv",
    r"C:\Users\ucwsh\Downloads\pseudomonas_aeruginosa.csv"
]

# 1.2 โฟลเดอร์ปลายทาง (Destination Folder) - สำหรับเก็บผลลัพธ์และอ่าน Mapping
DEST_FOLDER = os.path.join("MDR", "DrugClass2")

# 1.3 ไฟล์มาตรฐาน (Mapping & ID) - สมมติว่าวางอยู่ใน folder ปลายทางนี้
FILE_MAPPING = os.path.join(DEST_FOLDER, "Drug_class_for_MDR_new.csv")
FILE_ID = os.path.join(DEST_FOLDER, "ID.csv")

# ==============================================================================
# 2. ฟังก์ชันช่วยงาน (HELPER FUNCTIONS)
# ==============================================================================

def get_short_name(filename):
    """แปลงชื่อไฟล์ยาวๆ ให้เป็นชื่อสั้นๆ"""
    name = filename.replace('.csv', '')
    parts = name.split('_')
    if len(parts) >= 2:
        short_name = f"{parts[0][0].upper()}_{parts[1]}"
        return short_name
    return name

def normalize_string(text):
    """จัดรูปแบบข้อความรายการยา: แยก comma -> ตัดช่องว่าง -> เรียง A-Z -> รวมกลับ"""
    if pd.isna(text) or str(text).strip() == "" or str(text).lower() == 'nan':
        return ""
    parts = [p.strip() for p in str(text).split(',')]
    parts.sort()
    return ", ".join(parts)

# ==============================================================================
# 3. MAIN LOGIC
# ==============================================================================

def main():
    print(f"--- เริ่มต้นกระบวนการ (อ่านจาก Downloads -> บันทึกลง {DEST_FOLDER}) ---")
    
    # ตรวจสอบว่าโฟลเดอร์ปลายทางมีอยู่จริงไหม ถ้าไม่มีให้สร้าง
    if not os.path.exists(DEST_FOLDER):
        try:
            os.makedirs(DEST_FOLDER)
            print(f">> สร้างโฟลเดอร์: {DEST_FOLDER}")
        except Exception as e:
            print(f"[Error] ไม่สามารถสร้างโฟลเดอร์ปลายทางได้: {e}")
            return

    # --- STEP 3.1: โหลดไฟล์มาตรฐาน (Mapping & ID) ---
    if not os.path.exists(FILE_MAPPING) or not os.path.exists(FILE_ID):
        print(f"[Critical Error] ไม่พบไฟล์ Mapping หรือ ID ใน {DEST_FOLDER}")
        print("กรุณานำไฟล์ Drug_class_for_MDR_new.csv และ ID.csv ไปวางในโฟลเดอร์ MDR/DrugClass2 ก่อนครับ")
        return

    print(">> กำลังโหลดไฟล์มาตรฐาน (Mapping & ID)...")
    try:
        df_map = pd.read_csv(FILE_MAPPING, encoding='utf-8')
        df_id = pd.read_csv(FILE_ID, encoding='utf-8')
    except Exception as e:
        print(f"Error loading standard files: {e}")
        return

    # Mapping Data Preparation
    df_map.columns = df_map.columns.str.strip()
    organism_drug_map = {}
    for _, row in df_map.iterrows():
        org = str(row['ORGANISM_WHO']).strip().lower()
        drug = str(row['Antibiotic']).strip().lower()
        cls = str(row['Class']).strip()
        organism_drug_map[(org, drug)] = cls

    # ID Data Preparation
    df_id.columns = df_id.columns.str.strip()
    if 'std_tested_list' in df_id.columns:
        # แก้ไข Î² -> β
        df_id['std_tested_list'] = df_id['std_tested_list'].astype(str).str.replace("Î²-lactam", "β-lactam", regex=False)
    
    df_id['join_key'] = df_id['std_tested_list'].apply(normalize_string)
    print(">> โหลดไฟล์มาตรฐานเรียบร้อย\n")

    # --- STEP 3.2: วนลูปทำทีละไฟล์จาก Downloads ---
    for input_path in SOURCE_FILES:
        # ดึงชื่อไฟล์ออกมาจาก Path เต็ม (เช่น 'escherichia_coli.csv')
        filename = os.path.basename(input_path)
        
        # ตั้งชื่อไฟล์ Output และ Path ปลายทาง
        short_name = get_short_name(filename)
        output_filename = f"MDR_{short_name}.csv"
        output_path = os.path.join(DEST_FOLDER, output_filename)
        
        print(f"Processing: {filename} ...")
        
        if not os.path.exists(input_path):
            print(f"   [Skip] ไม่พบไฟล์ต้นทางที่: {input_path}")
            continue

        try:
            # 1. อ่านไฟล์เชื้อจาก Downloads
            df = pd.read_csv(input_path, encoding='utf-8')
            df.columns = df.columns.str.strip()

            if 'organism_full' not in df.columns or 'x_year' not in df.columns:
                print(f"   [Error] ขาดคอลัมน์สำคัญ")
                continue
            
            df['organism_full'] = df['organism_full'].astype(str).str.strip()

            # 2. หา Resistance Class
            def get_resistance(row):
                found = set()
                curr_org = str(row['organism_full']).lower()
                for col in df.columns:
                    if col.lower() in ['sample_id', 'organism_full', 'x_year']: continue
                    key = (curr_org, col.strip().lower())
                    if key in organism_drug_map:
                        if str(row[col]).strip().lower() == 'r':
                            found.add(organism_drug_map[key])
                if not found: return ""
                return ", ".join(sorted(list(found)))

            df['Resistant_Drug_Classes'] = df.apply(get_resistance, axis=1)

            # 3. กรอง MDR (>= 3 Classes) เพื่อหาตัวตั้ง
            df['Class_Count'] = df['Resistant_Drug_Classes'].apply(lambda x: len(x.split(',')) if x else 0)
            
            mdr_df = df[df['Class_Count'] >= 3].copy()

            if mdr_df.empty:
                print(f"   [Info] {filename} ไม่พบเคส MDR เลย -> ข้าม")
                continue

            # 4. สรุปยอด
            # 4.1 Count (Numerator): จาก mdr_df
            summary = mdr_df.groupby(['x_year', 'organism_full', 'Resistant_Drug_Classes']).size().reset_index(name='Count')
            
            # 4.2 Total (Denominator): จาก df (ข้อมูลทั้งหมด) **ตามที่ขอ**
            total_all = df.groupby(['x_year', 'organism_full']).size().reset_index(name='Total_Samples')
            
            # Merge & Calc %
            summary = pd.merge(summary, total_all, on=['x_year', 'organism_full'])
            summary['Percentage'] = (summary['Count'] / summary['Total_Samples'] * 100).round(2)
            summary = summary.sort_values(['x_year', 'Count'], ascending=[True, False])

            # 5. จับคู่กับ ID
            summary['join_key'] = summary['Resistant_Drug_Classes'].apply(normalize_string)
            final_df = pd.merge(summary, df_id[['join_key', 'MDR_id']], on='join_key', how='left')
            final_df.drop(columns=['join_key'], inplace=True)
            
            # จัดลำดับคอลัมน์
            cols_order = ['x_year', 'organism_full', 'Resistant_Drug_Classes', 'Count', 'Total_Samples', 'Percentage', 'MDR_id']
            final_cols = [c for c in cols_order if c in final_df.columns]
            final_df = final_df[final_cols]

            # 6. บันทึกไฟล์ลง MDR/DrugClass2
            final_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"   ✅ Saved to: {output_path}")

        except Exception as e:
            print(f"   [Error] เกิดข้อผิดพลาดกับไฟล์ {filename}: {e}")

    print("\n--- เสร็จสิ้นทุกขั้นตอน ---")

if __name__ == "__main__":
    main()