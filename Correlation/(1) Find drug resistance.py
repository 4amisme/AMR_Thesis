import pandas as pd
import os

def main():
    # --- 1. CONFIG PATH ---
    # ไฟล์หลักจากโฟลเดอร์ Downloads
    input_file_main = r"C:\Users\MP1YXGGZ\Downloads\AllYears_DrugClass_tested.csv"
    
    # ไฟล์ Mapping (สมมติว่าอยู่ที่เดียวกับไฟล์หลัก หรือโฟลเดอร์ที่โค้ดรู้จัก)
    # หากไฟล์นี้อยู่ที่อื่น ให้เปลี่ยน Path ตรงนี้ครับ
    mapping_folder = os.path.join("MDR", "DrugClass2")
    input_file_mapping = os.path.join(mapping_folder, "Drug_class_for_MDR_new.csv")
    
    # ชื่อไฟล์ Output สำหรับบันทึกผล
    output_path = r"C:\Users\MP1YXGGZ\Downloads\AllYears_Resistance_Full_Analysis.csv"

    print("--- เริ่มต้นกระบวนการ (วิเคราะห์ข้อมูลทั้งหมด | Missing_Count = 0) ---")

    # --- 2. LOAD FILES ---
    try:
        df_main = pd.read_csv(input_file_main, encoding='utf-8')
        df_mapping = pd.read_csv(input_file_mapping, encoding='utf-8')
        print(f"✅ โหลดไฟล์สำเร็จ: ข้อมูลหลัก {len(df_main)} แถว")
    except Exception as e:
        print(f"❌ [Error] ไม่สามารถโหลดไฟล์ได้: {e}")
        return

    # --- 3. CLEANING & EARLY FILTER (กรอง Missing_Count ก่อนเพื่อน) ---
    
    # 3.1 แปลงชื่อคอลัมน์เป็นตัวเล็กและตัดช่องว่างเพื่อป้องกัน Error
    df_main.columns = df_main.columns.str.strip().str.lower()
    df_mapping.columns = df_mapping.columns.str.strip().str.lower()

    # 3.2 กรองเฉพาะแถวที่ไม่มีข้อมูลขาดหาย (Missing_Count == 0)
    if 'missing_count' in df_main.columns:
        df_main = df_main[df_main['missing_count'] == 0].copy()
        print(f"✅ กรอง Missing_Count = 0 เรียบร้อย: เหลือ {len(df_main)} แถว")
    else:
        print("⚠️ ไม่พบคอลัมน์ 'missing_count' ข้ามการกรองส่วนนี้")

    # 3.3 ตรวจสอบคอลัมน์ชื่อเชื้อ
    if 'organism_full' not in df_main.columns:
        print(f"❌ ไม่พบคอลัมน์ 'organism_full' คอลัมน์ที่มีคือ: {list(df_main.columns)}")
        return

    # --- 4. MAPPING PREPARATION ---
    # สร้าง Dictionary เพื่อการ Lookup ที่รวดเร็ว
    organism_drug_map = {}
    for _, row in df_mapping.iterrows():
        org = str(row['organism_who']).strip().lower()
        drug = str(row['antibiotic']).strip().lower()
        cls = str(row['class']).strip()
        organism_drug_map[(org, drug)] = cls

    # --- 5. RESISTANCE ANALYSIS (วิเคราะห์ทุกแถว) ---
    
    # คัดเลือกคอลัมน์ที่เป็นรายชื่อยา (เอาคอลัมน์ที่ไม่เกี่ยวข้องออก)
    exclude_cols = ['sample_id', 'organism_full', 'x_year', 'missing_count', 'class_count', 'resistant_drug_classes']
    drug_cols = [c for c in df_main.columns if c not in exclude_cols]

    def analyze_resistance(row):
        found_classes = set()
        curr_org = str(row['organism_full']).strip().lower()
        
        for col in drug_cols:
            key = (curr_org, col)
            # ถ้ามีคู่นี้ในแผนผัง Mapping
            if key in organism_drug_map:
                val = str(row[col]).strip().lower()
                # เช็คว่าผลเป็นดื้อยา (R, Resistant หรือ 1)
                if val in ['r', 'resistant', '1']: 
                    found_classes.add(organism_drug_map[key])
        
        # คืนค่า (จำนวนกลุ่มยา, รายชื่อกลุ่มยา)
        count = len(found_classes)
        names = ", ".join(sorted(list(found_classes))) if found_classes else "-"
        return count, names

    print("\n[Processing] กำลังวิเคราะห์รูปแบบการดื้อยา...")
    
    # ประมวลผลและสร้างคอลัมน์ใหม่
    results = df_main.apply(analyze_resistance, axis=1)
    df_main['class_count'] = [x[0] for x in results]
    df_main['resistant_drug_classes'] = [x[1] for x in results]

    # --- 6. SAVE & SUMMARY ---
    # ไม่มีการ Filter Class_Count >= 3 เพื่อเก็บข้อมูลทั้งหมดไว้
    try:
        df_main.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✅ บันทึกไฟล์เรียบร้อยที่: {output_path}")
        
        # แสดงสรุปผลสั้นๆ
        print("-" * 50)
        print(f"สรุปผลการวิเคราะห์:")
        print(f"- จำนวนข้อมูลทั้งหมด: {len(df_main)} แถว")
        print(f"- ตัวอย่างข้อมูล:")
        print(df_main[['organism_full', 'class_count', 'resistant_drug_classes']].head(10))
        print("-" * 50)
    except Exception as e:
        print(f"❌ [Error] ไม่สามารถบันทึกไฟล์ได้: {e}")

if __name__ == "__main__":
    main()