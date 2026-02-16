import pandas as pd
import os

def main():
    # --- 1. CONFIG PATH ---
    mapping_folder = os.path.join("MDR", "DrugClass2")
    input_file_mapping = os.path.join(mapping_folder, "Drug_class_for_MDR_new.csv")

    input_folder = os.path.join("MDR", "DrugClass1.2")
    input_file_main = os.path.join(input_folder, "AllYears_DrugClass_tested.csv")
    
    output_path = os.path.join(input_folder, "AllYears_MDR_Pattern.csv")

    print("--- เริ่มต้นกระบวนการ (เวอร์ชั่นแก้ไขชื่อเชื้อ/คอลัมน์) ---")

    # --- 2. LOAD FILES ---
    try:
        df_main = pd.read_csv(input_file_main, encoding='utf-8')
        df_mapping = pd.read_csv(input_file_mapping, encoding='utf-8')
        print(f"✅ โหลดไฟล์สำเร็จ")
    except Exception as e:
        print(f"[Error] {e}")
        return

    # --- 3. CLEANING (จุดสำคัญที่แก้ปัญหา) ---
    
    # 3.1 แปลงชื่อคอลัมน์เป็นตัวเล็กทั้งหมด และตัดช่องว่าง (แก้ปัญหา Organism_full vs organism_full)
    df_main.columns = df_main.columns.str.strip().str.lower()
    df_mapping.columns = df_mapping.columns.str.strip().str.lower()
    
    print(f"   - คอลัมน์ในไฟล์ข้อมูล (ตัวอย่าง): {list(df_main.columns)[:5]}")

    # 3.2 ตรวจสอบชื่อเชื้อ (Organism Check)
    if 'organism_full' not in df_main.columns:
        print("❌ ไม่พบคอลัมน์ 'organism_full' (หลังจากแปลงเป็นตัวเล็กแล้ว)")
        print(f"คอลัมน์ที่มีคือ: {list(df_main.columns)}")
        return

    # ดึงชื่อเชื้อที่มีในข้อมูลจริง
    data_organisms = set(df_main['organism_full'].astype(str).str.strip().str.lower().unique())
    # ดึงชื่อเชื้อที่มีใน Mapping
    map_organisms = set(df_mapping['organism_who'].astype(str).str.strip().str.lower().unique())

    # หาตัวที่ตรงกัน
    matched_orgs = data_organisms.intersection(map_organisms)
    
    print("\n[ตรวจสอบชื่อเชื้อ]")
    print(f"   - เชื้อในไฟล์ข้อมูล: {len(data_organisms)} ชนิด")
    print(f"   - เชื้อใน Mapping: {len(map_organisms)} ชนิด")
    print(f"   - **ตรงกัน**: {len(matched_orgs)} ชนิด")

    if len(matched_orgs) == 0:
        print("🔴 ปัญหาพบแล้ว: ชื่อเชื้อเขียนไม่เหมือนกันเลย!")
        print(f"   - ตัวอย่างในไฟล์ข้อมูล: {list(data_organisms)[:3]}")
        print(f"   - ตัวอย่างใน Mapping: {list(map_organisms)[:3]}")
        return

    # --- 4. FILTER Missing_Count = 0 ---
    target_col = 'missing_count' # เป็นตัวเล็กแล้ว
    if target_col in df_main.columns:
        print(f"\n[Filter] กรอง {target_col} == 0 ...")
        df_main = df_main[df_main[target_col] == 0].copy()
        print(f"   - เหลือข้อมูล: {len(df_main)} แถว")
    else:
        print(f"⚠️ ไม่พบ {target_col} ข้ามการกรองนี้")

    # --- 5. MAPPING PREP ---
    print("\n[Processing] กำลังหา MDR...")
    
    organism_drug_map = {}
    for _, row in df_mapping.iterrows():
        # key = (เชื้อตัวเล็ก, ยาตัวเล็ก)
        org = str(row['organism_who']).strip().lower()
        drug = str(row['antibiotic']).strip().lower()
        cls = str(row['class']).strip()
        organism_drug_map[(org, drug)] = cls

    # --- 6. FIND RESISTANCE ---
    # ใช้ dictionary lookup เพื่อความเร็ว (แทน apply)
    
    # 1. สร้าง key ยา ทั้งหมดใน df รอไว้
    drug_cols = [c for c in df_main.columns if c not in ['sample_id', 'organism_full', 'x_year', 'missing_count', 'resistant_drug_classes']]
    
    def get_mdr_classes(row):
        found = set()
        # เชื้อของแถวนี้
        curr_org = str(row['organism_full']).strip().lower()
        
        for col in drug_cols:
            # สร้าง key (เชื้อ, ยา)
            key = (curr_org, col)
            
            # ถ้าคู่นี้มีใน mapping และ ผลเป็น 'r'
            if key in organism_drug_map:
                val = str(row[col]).strip().lower()
                # เช็ค R หรือ Resistant หรือ 1
                if val in ['r', 'resistant', '1']: 
                    found.add(organism_drug_map[key])
        
        if not found: return 0
        return len(found)

    # คำนวณจำนวน Class ทันที (ไม่ต้องเก็บ text เพื่อประหยัดเมม)
    # *วิธีนี้ช้าหน่อยแต่ชัวร์ ถ้าข้อมูลเยอะมากอาจรอนานนิดนึง*
    df_main['Class_Count'] = df_main.apply(get_mdr_classes, axis=1)

    # --- 7. FINAL FILTER & SAVE ---
    mdr_df = df_main[df_main['Class_Count'] >= 3].copy()
    
    print(f"\n[Result] พบ MDR (>= 3 classes): {len(mdr_df)} แถว")

    if not mdr_df.empty:
        # ถ้าอยากได้รายชื่อยาด้วย (Optional) ให้ gen ใหม่เฉพาะตัวที่ผ่าน
        # (ทำตรงนี้เพื่อประหยัดเวลาตอน loop แรก)
        def get_class_names(row):
            found = set()
            curr_org = str(row['organism_full']).strip().lower()
            for col in drug_cols:
                key = (curr_org, col)
                if key in organism_drug_map and str(row[col]).strip().lower() in ['r', 'resistant', '1']:
                    found.add(organism_drug_map[key])
            return ", ".join(sorted(list(found)))
            
        mdr_df['Resistant_Drug_Classes'] = mdr_df.apply(get_class_names, axis=1)
        
        # บันทึก
        mdr_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ บันทึกเสร็จสมบูรณ์: {output_path}")
        print(mdr_df[['organism_full', 'Class_Count', 'Resistant_Drug_Classes']].head())
    else:
        print("❌ ยังไม่เจอ MDR อีก... (ต้องเช็คชื่อเชื้อที่ print ด้านบนว่าตรงกันไหม)")

if __name__ == "__main__":
    main()