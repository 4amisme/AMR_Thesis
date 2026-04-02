import pandas as pd
import os

def debug_mdr():
    # --- ตั้งค่า Path ---
    base_folder = os.path.join("MDR", "DrugClass2")
    mapping_path = os.path.join(base_folder, "Drug_class_for_MDR_new.csv")
    
    input_folder = os.path.join("MDR", "DrugClass1.2")
    data_path = os.path.join(input_folder, "AllYears_DrugClass_tested.csv")

    print("--- เริ่มการตรวจสอบ (Debug Mode) ---")

    # 1. โหลดข้อมูล
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
        df_map = pd.read_csv(mapping_path, encoding='utf-8')
        print(f"✅ อ่านไฟล์สำเร็จ")
        print(f"   - ข้อมูลทั้งหมด: {len(df)} แถว")
    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่สำเร็จ: {e}")
        return

    # 2. เช็ค Missing_Count
    target_col = 'Missing_Count' # หรือ 'Missing_count' เช็คให้ดี
    if target_col in df.columns:
        zeros = df[df[target_col] == 0]
        print(f"\n[เช็ค 1] การกรอง {target_col} == 0")
        print(f"   - จำนวนแถวที่เหลือ: {len(zeros)}")
        if len(zeros) == 0:
            print("   🔴 ปัญหาเจอแล้ว! -> ไม่มีแถวไหนเลยที่มี Missing_Count เป็น 0")
            print("   -> แนะนำ: ลองเช็คไฟล์ CSV ว่าคอลัมน์นี้มีค่าอะไรบ้าง")
            print(f"   -> ตัวอย่างค่าที่มี: {df[target_col].unique()[:5]}")
            return # จบการทำงานเพราะไม่มีข้อมูลให้ไปต่อ
        else:
            df = zeros # ใช้ข้อมูลที่กรองแล้วไปเช็คต่อ
    else:
        print(f"   ⚠️ ไม่พบคอลัมน์ {target_col} (ข้ามขั้นตอนนี้)")

    # 3. เช็คชื่อยา (Mapping Matching)
    print(f"\n[เช็ค 2] การจับคู่ชื่อยา (Mapping)")
    # เตรียม Mapping set
    map_drugs = set(df_map['Antibiotic'].astype(str).str.strip().str.lower().unique())
    # เตรียม Column set
    col_drugs = set(x.strip().lower() for x in df.columns)
    
    # หาตัวที่ตรงกัน (Intersection)
    matched = map_drugs.intersection(col_drugs)
    print(f"   - ชื่อยาใน Mapping มี: {len(map_drugs)} ตัว")
    print(f"   - ชื่อคอลัมน์ในไฟล์ข้อมูล มี: {len(col_drugs)} ตัว")
    print(f"   - **จับคู่ชื่อยาตรงกันได้**: {len(matched)} ตัว")
    
    if len(matched) == 0:
        print("   🔴 ปัญหาเจอแล้ว! -> ชื่อยาไม่ตรงกันเลยแม้แต่ตัวเดียว")
        print("   -> ตัวอย่างใน Mapping: ", list(map_drugs)[:3])
        print("   -> ตัวอย่างในไฟล์ข้อมูล: ", list(col_drugs)[:3])
        return

    # 4. เช็คค่า 'R' (Value Check)
    print(f"\n[เช็ค 3] ค่าความเป็น R ในตาราง")
    # ลองสุ่มคอลัมน์ยาที่แมพเจอมา 1 ตัว
    test_drug = list(matched)[0] 
    
    # หาชื่อคอลัมน์จริง (Original Case)
    original_col = [c for c in df.columns if c.strip().lower() == test_drug][0]
    
    unique_vals = df[original_col].unique()
    print(f"   - ลองตรวจสอบคอลัมน์ '{original_col}'")
    print(f"   - ค่าที่พบในคอลัมน์นี้คือ: {unique_vals}")
    
    has_r = any(str(v).strip().lower() == 'r' for v in unique_vals)
    if not has_r:
        print(f"   🔴 ปัญหาเจอแล้ว! -> ในคอลัมน์นี้ไม่มีค่า 'r' หรือ 'R' เลย")
        print("   -> (โปรแกรมอาจเห็นเป็น 'Resistant', '1', หรืออื่นๆ ต้องแก้โค้ด)")
    else:
        print(f"   ✅ พบค่าน่าสงสัยว่าเป็น 'R' (Logic ตรงนี้น่าจะผ่าน)")

    print("\n--- จบการตรวจสอบ ---")

if __name__ == "__main__":
    debug_mdr()