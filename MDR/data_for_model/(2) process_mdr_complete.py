import pandas as pd
import os

def run_strict_process():
    # ==========================================
    # ⚙️ 1. ตั้งค่าไฟล์
    # ==========================================    
    input_file = os.path.join("MDR", "data_for_model", "AllYears_processed.csv")
    mapping_file = os.path.join("MDR", "data_for_model", "Drug_class_for_MDR.csv")
    output_file = os.path.join("MDR", "data_for_model", "AllYears_MDR_Final.csv")

    print(f"🚀 เริ่มกระบวนการประมวลผล (โหมดคัดกรองเข้มข้น)...")

    # ตรวจสอบไฟล์
    if not os.path.exists(input_file):
        print(f"❌ Error: ไม่พบไฟล์ {input_file}")
        return
    if not os.path.exists(mapping_file):
        print(f"❌ Error: ไม่พบไฟล์ {mapping_file}")
        return

    # อ่านไฟล์
    try:
        df_main = pd.read_csv(input_file, low_memory=False)
        df_map = pd.read_csv(mapping_file)
    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่สำเร็จ: {e}")
        return

    # ==========================================
    # 🧹 2. ทำความสะอาดข้อมูล
    # ==========================================
    print("🧹 กำลังทำความสะอาดชื่อคอลัมน์...")
    df_main.columns = df_main.columns.str.strip().str.lower()
    df_map.columns = df_map.columns.str.strip().str.lower()

    # หาชื่อคอลัมน์ในไฟล์ Mapping
    ab_col = next((c for c in df_map.columns if c in ['antibiotic', 'drug', 'name']), None)
    class_col = next((c for c in df_map.columns if c in ['class', 'drug_class']), None)

    if not ab_col or not class_col:
        print("❌ ไฟล์ Mapping ขาดหัวตาราง Antibiotic หรือ Class")
        return
    
    # Clean ข้อมูลใน Mapping
    df_map[ab_col] = df_map[ab_col].astype(str).str.strip().str.lower()
    df_map[class_col] = df_map[class_col].astype(str).str.strip()

    # 🌟 จุดสำคัญ 1: สร้าง "ทำเนียบชื่อ Class ที่ถูกต้อง" (Whitelist)
    # โปรแกรมจะจำไว้ว่าชื่อไหนบ้างที่เป็นชื่อ Class จริงๆ (เช่น Aminoglycosides)
    valid_known_classes = set(df_map[class_col].unique())
    print(f"ℹ️ ระบบจดจำชื่อ Class ที่ถูกต้องได้ทั้งหมด {len(valid_known_classes)} ชื่อ")

    # รายชื่อยาที่เราจะตรวจสอบในไฟล์หลัก
    target_drugs = [d for d in df_map[ab_col] if d in df_main.columns]

    # ==========================================
    # 🔄 3. เปลี่ยนค่า R เป็นชื่อ Class
    # ==========================================
    print("🔄 [Step 1] กำลังเปลี่ยนค่า R เป็นชื่อ Class...")
    
    for index, row in df_map.iterrows():
        drug_name = row[ab_col]
        drug_class = row[class_col]
        
        if drug_name in df_main.columns:
            # เงื่อนไข: เป็น R หรือ r
            condition = df_main[drug_name].astype(str).str.strip().str.upper() == 'R'
            if condition.sum() > 0:
                df_main.loc[condition, drug_name] = drug_class

    # ==========================================
    # ➕ 4. รวมกลุ่มยา (ใช้ระบบ Whitelist)
    # ==========================================
    print("➕ [Step 2] กำลังกรองและรวมชื่อ Class (ตัดค่า ?, NS, SDD ทิ้ง)...")

    def combine_valid_classes(row):
        found_classes = set() 
        for col in target_drugs:
            # ดึงค่าออกมา (อาจจะเป็น Aminoglycosides, S, I, ?, NS, SDD)
            val = str(row[col]).strip()
            
            # 🌟 จุดสำคัญ 2: ตรวจสอบว่าคำนี้ "มีอยู่ในทำเนียบ Class" หรือไม่?
            # ถ้าใช่ (เช่นเป็น Aminoglycosides) -> เก็บ
            # ถ้าไม่ใช่ (เช่นเป็น ?, NS, SDD, S, I) -> ทิ้งทันที
            if val in valid_known_classes:
                found_classes.add(val)
        
        return ", ".join(sorted(found_classes)) if found_classes else ""

    df_main['drug_classes'] = df_main.apply(combine_valid_classes, axis=1)

    # ==========================================
    # ⚖️ 5. ตัดสินผล MDR
    # ==========================================
    print("⚖️ [Step 3] กำลังตรวจสอบสถานะ MDR...")

    def check_mdr(val_str):
        if not val_str: return 'S'
        
        classes_list = [c for c in val_str.split(',') if c.strip()]
        
        # >= 3 กลุ่ม -> R (MDR)
        if len(classes_list) >= 3:
            return 'R'
        else:
            return 'S'

    df_main['MDR'] = df_main['drug_classes'].apply(check_mdr)

    # ==========================================
    # 💾 6. บันทึกไฟล์
    # ==========================================
    print(f"💾 กำลังบันทึกไฟล์ผลลัพธ์: {output_file}")
    df_main.to_csv(output_file, index=False)
    
    print("-" * 60)
    print("✅ เสร็จสมบูรณ์! ขยะข้อมูล (?, NS, SDD) ถูกกรองออกเรียบร้อย")
    print("ตัวอย่างข้อมูล:")
    print(df_main[['organism', 'drug_classes', 'MDR']].head(10))
    print("-" * 60)

if __name__ == "__main__":
    run_strict_process()