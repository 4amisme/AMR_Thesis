import pandas as pd
import os

def main():
    # --- 1. ตั้งค่า Path ---
    base_folder = os.path.join("MDR", "DrugClass2")
    input_file_main = os.path.join(base_folder, "acinetobacter_baumannii.csv")
    input_file_mapping = os.path.join(base_folder, "Drug_class_for_MDR_new.csv")
    
    # ชื่อไฟล์ผลลัพธ์ (เปลี่ยนชื่อให้สื่อความหมายว่าเป็น MDR >= 3)
    output_detailed = os.path.join(base_folder, "acinetobacter_baumannii_MDR_filtered.csv")
    output_summary = os.path.join(base_folder, "summary_MDR_only_by_year.csv")

    if not os.path.exists(input_file_main) or not os.path.exists(input_file_mapping):
        print(f"[Error] ไม่พบไฟล์ข้อมูลใน: {base_folder}")
        return

    print("--- ขั้นตอนที่ 1: โหลดและประมวลผลข้อมูล ---")
    
    try:
        df_main = pd.read_csv(input_file_main, encoding='utf-8')
        df_mapping = pd.read_csv(input_file_mapping, encoding='utf-8')
    except Exception as e:
        print(f"[Error] อ่านไฟล์ไม่สำเร็จ: {e}")
        return

    # --- Data Cleaning ---
    df_main.columns = df_main.columns.str.strip()
    
    if 'x_year' not in df_main.columns:
        print("[Error] ไม่พบคอลัมน์ 'x_year'")
        return

    # Prepare Mapping
    df_mapping['key_organism'] = df_mapping['ORGANISM_WHO'].astype(str).str.strip().str.lower()
    df_mapping['key_antibiotic'] = df_mapping['Antibiotic'].astype(str).str.strip().str.lower()
    df_mapping['Class'] = df_mapping['Class'].astype(str).str.strip()

    organism_drug_map = {}
    for _, row in df_mapping.iterrows():
        key = (row['key_organism'], row['key_antibiotic'])
        organism_drug_map[key] = row['Class']

    # --- Processing Function ---
    def find_resistant_classes(row):
        found_classes = set()
        if 'organism_full' not in row: return ""
            
        current_organism = str(row['organism_full']).strip().lower()
        
        for col_name in df_main.columns:
            if col_name.lower() in ['sample_id', 'organism_full', 'resistant_drug_classes', 'x_year']:
                continue
            
            lookup_key = (current_organism, col_name.strip().lower())
            
            if lookup_key in organism_drug_map:
                val = str(row[col_name]).strip().lower()
                if val == 'r':
                    found_classes.add(organism_drug_map[lookup_key])
        
        if not found_classes:
            return "" # ถ้าไม่เจอให้เป็นว่างไว้ก่อน เพื่อให้นับจำนวนได้ง่าย
        
        return ", ".join(sorted(list(found_classes)))

    # Apply function
    df_main['Resistant_Drug_Classes'] = df_main.apply(find_resistant_classes, axis=1)

    # ==============================================================================
    # --- ขั้นตอนที่ 2: กรองเฉพาะ MDR (>= 3 Classes) ---
    # ==============================================================================
    print("\n--- ขั้นตอนที่ 2: กรองข้อมูลเฉพาะ MDR (>= 3 classes) ---")

    # ฟังก์ชันนับจำนวน Class
    def count_classes(text):
        if not text or text == "":
            return 0
        # แยกด้วย comma แล้วนับจำนวน
        return len(text.split(','))

    # สร้างคอลัมน์ช่วยนับจำนวน
    df_main['Class_Count'] = df_main['Resistant_Drug_Classes'].apply(count_classes)

    # *** FILTER: เลือกเฉพาะแถวที่มี Class_Count >= 3 ***
    mdr_df = df_main[df_main['Class_Count'] >= 3].copy()
    
    print(f"จำนวนข้อมูลทั้งหมด: {len(df_main)} แถว")
    print(f"จำนวนข้อมูลที่เป็น MDR (>=3 classes): {len(mdr_df)} แถว")

    # บันทึกไฟล์รายละเอียด (เฉพาะ MDR)
    mdr_df.to_csv(output_detailed, index=False, encoding='utf-8')
    print(f"✅ บันทึกไฟล์รายชื่อ MDR เรียบร้อยที่: {output_detailed}")

    if mdr_df.empty:
        print("[Warning] ไม่พบข้อมูลที่ดื้อยาตั้งแต่ 3 กลุ่มขึ้นไปเลย โปรแกรมจะหยุดทำงาน")
        return

    # ==============================================================================
    # --- ขั้นตอนที่ 3: สรุปผล MDR ตามปี ---
    # ==============================================================================
    print("\n--- ขั้นตอนที่ 3: คำนวณสัดส่วน % ของ MDR pattern ---")

    # 1. Group by ปี และ รูปแบบการดื้อยา (จากตารางที่กรองแล้ว)
    summary_df = mdr_df.groupby(['x_year', 'Resistant_Drug_Classes']).size().reset_index(name='Count')

    # 2. หาจำนวน MDR ทั้งหมดในแต่ละปี (MDR Total per Year)
    # หมายเหตุ: Percentage จะคิดจากฐานจำนวน MDR ไม่ใช่จำนวนผู้ป่วยทั้งหมด
    total_mdr_per_year = mdr_df.groupby('x_year').size().reset_index(name='Total_MDR_In_Year')

    # 3. Merge ข้อมูล
    summary_df = pd.merge(summary_df, total_mdr_per_year, on='x_year')

    # 4. คำนวณ %
    summary_df['Percentage'] = (summary_df['Count'] / summary_df['Total_MDR_In_Year']) * 100
    summary_df['Percentage'] = summary_df['Percentage'].round(2)

    # 5. เรียงลำดับ
    summary_df = summary_df.sort_values(by=['x_year', 'Count'], ascending=[True, False])

    # --- บันทึกไฟล์สรุป ---
    try:
        summary_df.to_csv(output_summary, index=False, encoding='utf-8')
        
        print("-" * 60)
        print(f"✅ บันทึกไฟล์สรุป MDR เรียบร้อย!")
        print(f"ไฟล์อยู่ที่: {output_summary}")
        print("-" * 60)
        
        # แสดงตัวอย่าง
        print("\nตัวอย่างตารางสรุป MDR (Top 10 rows):")
        pd.set_option('display.max_colwidth', 50)
        print(summary_df.head(10).to_string(index=False))

    except Exception as e:
        print(f"บันทึกไฟล์สรุปไม่สำเร็จ: {e}")

if __name__ == "__main__":
    main()