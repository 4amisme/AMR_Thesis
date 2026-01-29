import pandas as pd
import os

def add_tested_class_info():
    # ==========================================
    # 1. ตั้งค่าไฟล์
    # ==========================================
    input_file = os.path.join("MDR", "data", "AllYears_processed.csv")
    mapping_file = os.path.join("MDR", "data_some_isolate", "Drug_class_for_MDR.csv")
    
    # ไฟล์ผลลัพธ์ที่จะบันทึกใน path MDR/data_some_isolate
    output_path = os.path.join("MDR", "data_some_isolate")
    output_filename = "AllYears_Checked_Classes.csv"
    output_file = os.path.join(output_path, output_filename)

    print(f"🚀 เริ่มกระบวนการตรวจสอบ Class ที่ตรวจ (Row-by-Row Check)...")

    # ตรวจสอบไฟล์
    if not os.path.exists(input_file) or not os.path.exists(mapping_file):
        print("❌ Error: ไม่พบไฟล์ Input หรือ Mapping")
        return

    # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    os.makedirs(output_path, exist_ok=True)

    try:
        df_main = pd.read_csv(input_file, low_memory=False)
        df_map = pd.read_csv(mapping_file)
    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่สำเร็จ: {e}")
        return

    # ==========================================
    # 2. เตรียมข้อมูล (Preprocessing)
    # ==========================================
    print("🧹 กำลังเตรียม Mapping...")
    
    # 1. Standardize Column Names (ตัวเล็ก + ตัดช่องว่าง)
    df_main.columns = df_main.columns.str.strip().str.lower()
    df_map.columns = df_map.columns.str.strip().str.lower()

    # 2. หาชื่อคอลัมน์ใน Mapping
    ab_col = next((c for c in df_map.columns if c in ['antibiotic', 'drug', 'name']), None)
    class_col = next((c for c in df_map.columns if c in ['class', 'drug_class']), None)
    
    # 3. เตรียม Dictionary: Class -> [List of Drugs]
    # (เอาเฉพาะยาที่มีคอลัมน์อยู่จริงในไฟล์หลัก เพื่อไม่ให้ error)
    df_map[ab_col] = df_map[ab_col].str.strip().str.lower()
    df_map[class_col] = df_map[class_col].str.strip() # ชื่อ Class เก็บ Case เดิมไว้สวยกว่า

    all_classes = df_map[class_col].dropna().unique()
    class_to_drugs_map = {}

    for cls in all_classes:
        drugs = df_map[df_map[class_col] == cls][ab_col].tolist()
        available_drugs = [d for d in drugs if d in df_main.columns]
        
        if available_drugs:
            class_to_drugs_map[cls] = available_drugs

    print(f"ℹ️ พบ Class ที่สามารถตรวจสอบได้ทั้งหมด: {len(class_to_drugs_map)} Class")

    # ==========================================
    # 3. เริ่มตรวจสอบทีละแถว (Logic)
    # ==========================================
    print("🔍 กำลังสแกนข้อมูลและสร้างคอลัมน์ใหม่ (อาจใช้เวลาสักครู่)...")
    
    valid_values = {'R', 'S', 'I'}

    def get_tested_info(row):
        found_classes = []
        
        for cls, drugs in class_to_drugs_map.items():
            # เช็คว่ามียาตัวใดตัวหนึ่งใน Class นี้มีผล R/S/I หรือไม่
            is_tested = False
            for d in drugs:
                val = str(row[d]).strip().upper()
                if val in valid_values:
                    is_tested = True
                    break # เจอ 1 ตัวก็พอแล้วสำหรับ Class นี้
            
            if is_tested:
                found_classes.append(cls)
        
        # คืนค่ากลับไป 2 ค่า (จำนวน, รายชื่อ)
        return len(found_classes), ", ".join(sorted(found_classes))

    # ใช้ .apply เพื่อสร้าง 2 คอลัมน์ใหม่
    # result_series จะเป็น DataFrame ชั่วคราวที่มี 2 คอลัมน์
    result_series = df_main.apply(get_tested_info, axis=1, result_type='expand')
    
    # ตั้งชื่อคอลัมน์ใหม่
    df_main['count_tested_classes'] = result_series[0]
    df_main['list_tested_classes'] = result_series[1]

    # ==========================================
    # 4. บันทึกผลลัพธ์
    # ==========================================
    print(f"💾 กำลังบันทึกไฟล์ใหม่: {output_file}")
    df_main.to_csv(output_file, index=False)

    print("-" * 60)
    print("✅ เสร็จสมบูรณ์! เพิ่มคอลัมน์ใหม่เรียบร้อย:")
    print("   1. 'count_tested_classes' (จำนวน Class ที่ตรวจ)")
    print("   2. 'list_tested_classes' (รายชื่อ Class ที่ตรวจ)")
    print("-" * 60)
    print("ตัวอย่างข้อมูล:")
    print(df_main[['organism', 'count_tested_classes', 'list_tested_classes']].head(10))
    print("-" * 60)

if __name__ == "__main__":
    add_tested_class_info()