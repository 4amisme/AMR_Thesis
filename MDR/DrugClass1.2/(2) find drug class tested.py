import pandas as pd
import os

def main():
    # --- ส่วนที่ 1: กำหนด Path ---
    data_path = os.path.join("MDR", "data", "AllYears_processed.csv")
    mapping_path = os.path.join("MDR", "DrugClass1.2", "Drug_class_for_MDR_new.csv")
    output_dir = os.path.join("MDR", "DrugClass1.2")
    output_path = os.path.join(output_dir, "AllYears_DrugClass_tested.csv")

    if not os.path.exists(data_path) or not os.path.exists(mapping_path):
        print("❌ ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบ Path อีกครั้ง")
        return

    print("🚀 เริ่มต้นโหลดข้อมูลและทำความสะอาดชื่อคอลัมน์...")

    # --- ส่วนที่ 2: เตรียม Mapping มาตรฐาน ---
    df_map = pd.read_csv(mapping_path)
    # ทำให้เกณฑ์มาตรฐานสะอาดที่สุด: ตัดช่องว่าง และแปลงเป็นตัวใหญ่
    df_map['ORGANISM_WHO'] = df_map['ORGANISM_WHO'].astype(str).str.strip().str.upper()
    df_map['Antibiotic'] = df_map['Antibiotic'].astype(str).str.strip().str.upper()
    df_map['Class'] = df_map['Class'].astype(str).str.strip()
    
    standard_organisms = set(df_map['ORGANISM_WHO'].unique())

    # --- ส่วนที่ 3: โหลดไฟล์หลักและจัดการชื่อคอลัมน์ (The Fix) ---
    df_data = pd.read_csv(data_path, low_memory=False)

    # 🛠️ ขั้นตอนการล้างชื่อคอลัมน์ (Clean Column Names)
    # ตัดช่องว่างหน้า-หลัง และแปลงชื่อคอลัมน์ทุกตัวให้เป็นตัวพิมพ์ใหญ่ชั่วคราวเพื่อเช็ค
    # แต่เราจะยังรักษาชื่อเดิมไว้เพื่อบันทึกลงไฟล์
    original_cols = df_data.columns
    clean_cols_map = {col.strip().upper(): col for col in original_cols}
    
    # ระบุชื่อคอลัมน์เชื้อที่ต้องใช้ (รองรับทั้ง 'organism_full', 'ORGANISM_FULL' ฯลฯ)
    target_key = 'ORGANISM_FULL'
    if target_key not in clean_cols_map:
        print(f"❌ ไม่พบคอลัมน์ {target_key} (หลังจากการทำความสะอาดชื่อแล้ว)")
        print(f"คอลัมน์ที่คุณมีคือ: {list(original_cols)}")
        return
    
    actual_organism_col = clean_cols_map[target_key]

    # --- ส่วนที่ 4: กรองข้อมูลเฉพาะเชื้อที่ตรงกัน ---
    initial_count = len(df_data)
    # กรองโดยไม่สน Case และ Space
    df_data = df_data[df_data[actual_organism_col].astype(str).str.strip().str.upper().isin(standard_organisms)].copy()
    filtered_count = len(df_data)
    
    print(f"✅ กรองข้อมูลเสร็จสิ้น: จาก {initial_count} แถว เหลือ {filtered_count} แถว")

    # เตรียม Rules
    rules = {}
    for _, row in df_map.iterrows():
        org = row['ORGANISM_WHO']
        anti = row['Antibiotic']
        cls = row['Class']
        if org not in rules:
            rules[org] = {}
        rules[org][anti] = cls

    # --- ส่วนที่ 5: ฟังก์ชันวิเคราะห์ (รองรับความไม่แน่นอนของชื่อยา) ---
    def analyze_row(row):
        current_org = str(row[actual_organism_col]).strip().upper()
        org_rules = rules.get(current_org, {})
        required_classes = set(org_rules.values())
        
        tested_classes = set()

        for anti_std, class_name in org_rules.items():
            # เช็คว่าชื่อยาในมาตรฐาน (anti_std) มีอยู่ในชื่อคอลัมน์ที่คลีนแล้วหรือไม่
            if anti_std in clean_cols_map:
                actual_drug_col = clean_cols_map[anti_std]
                val = row[actual_drug_col]
                if pd.notna(val) and str(val).strip() not in ["", "-"]:
                    tested_classes.add(class_name)

        missing_classes = required_classes - tested_classes
        
        return pd.Series({
            'Tested_Classes': ", ".join(sorted(list(tested_classes))),
            'Missing_Classes': ", ".join(sorted(list(missing_classes))),
            'Missing_Count': len(missing_classes)
        })

    print("📊 กำลังวิเคราะห์ Class ยา...")
    analysis_results = df_data.apply(analyze_row, axis=1)
    df_final = pd.concat([df_data, analysis_results], axis=1)

    # บันทึกไฟล์
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_final.to_csv(output_path, index=False, encoding='utf-8')
    print(f"🎉 เสร็จสมบูรณ์! ไฟล์บันทึกที่: {output_path}")

if __name__ == "__main__":
    main()