import pandas as pd
import os

def check_missing_antibiotic():
    # 1. ตั้งค่าพาธไฟล์ (ให้ตรงกับที่คุณใช้)
    input_file = os.path.join("MDR", "data", "AllYears_processed.csv")
    mapping_file = os.path.join("MDR", "DrugClass", "Drug_class_for_MDR.csv")

    print("🔍 กำลังเริ่มตรวจสอบรายชื่อยาที่หายไป...")

    # 2. ตรวจสอบว่าไฟล์มีอยู่จริงไหม
    if not os.path.exists(input_file):
        print(f"❌ ไม่พบไฟล์ข้อมูลที่: {input_file}")
        return
    if not os.path.exists(mapping_file):
        print(f"❌ ไม่พบไฟล์ Mapping ที่: {mapping_file}")
        return

    # 3. อ่านข้อมูล
    # อ่านเฉพาะ Header ของไฟล์หลักเพื่อความรวดเร็ว
    df_main_cols = pd.read_csv(input_file, nrows=0).columns.str.strip().str.lower().tolist()
    df_map = pd.read_csv(mapping_file)

    # 4. เตรียมชื่อคอลัมน์ใน Mapping
    df_map.columns = df_map.columns.str.strip().str.lower()
    ab_col = next((c for c in df_map.columns if c in ['antibiotic', 'drug', 'name']), None)

    if not ab_col:
        print("❌ ไม่พบคอลัมน์ 'antibiotic' ในไฟล์ Mapping")
        return

    # 5. ดึงรายชื่อยาจาก Mapping (Standardized)
    mapping_drugs = df_map[ab_col].dropna().astype(str).str.strip().str.lower().unique().tolist()
    mapping_drugs.sort()

    # 6. เปรียบเทียบ
    found_drugs = [d for d in mapping_drugs if d in df_main_cols]
    missing_drugs = [d for d in mapping_drugs if d not in df_main_cols]

    # 7. แสดงผลลัพธ์
    print("\n" + "="*50)
    print(f"📊 สรุปการตรวจสอบ:")
    print(f"   - ยาทั้งหมดใน Mapping: {len(mapping_drugs)} ชนิด")
    print(f"   - ยาที่หาเจอในไฟล์หลัก: {len(found_drugs)} ชนิด")
    print(f"   - ❌ ยาที่หา 'ไม่เจอ' ({len(missing_drugs)} ชนิด)")
    print("="*50)

    if missing_drugs:
        print("\nรายชื่อยาที่หาไม่เจอ:")
        for i, m in enumerate(missing_drugs, 1):
            print(f"{i}. {m}")
            
            # ตรวจสอบหาชื่อที่ใกล้เคียงในไฟล์หลัก
            similar = [c for c in df_main_cols if m in c or c in m]
            if similar:
                print(f"   💡 พบชื่อที่ใกล้เคียงในไฟล์หลัก: {similar}")
            else:
                print(f"   💡 ไม่พบชื่อที่ใกล้เคียงเลย (โปรดตรวจสอบการสะกดคำ)")
    else:
        print("\n✅ ยาทุกตัวใน Mapping มีคอลัมน์ตรงกับไฟล์หลักทั้งหมด!")

    print("\n" + "="*50)

if __name__ == "__main__":
    check_missing_antibiotic()