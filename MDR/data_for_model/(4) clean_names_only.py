import pandas as pd
import os

def clean_names_no_calc():
    # ชื่อไฟล์
    target_file = os.path.join("MDR", "data_for_model", "AllYears_MDR_for_model.csv")
    
    print(f"📂 กำลังอ่านไฟล์: {target_file}")

    if not os.path.exists(target_file):
        print(f"❌ Error: ไม่พบไฟล์ {target_file}")
        return

    try:
        df = pd.read_csv(target_file, low_memory=False)
    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่ได้: {e}")
        return

    # ==========================================
    # 1. เช็คก่อนแก้ (Before)
    # ==========================================
    original_names = df['organism_full'].unique()
    print(f"📊 จำนวนชื่อเชื้อก่อนแก้: {len(original_names)} ชื่อ")
    
    # ==========================================
    # 2. ปฏิบัติการแก้ชื่อ (Text Cleaning Only)
    # ==========================================
    print("🛠️ กำลังลบช่องว่างส่วนเกินออกจากชื่อ (ไม่คำนวณตัวเลขใหม่)...")
    
    # 1. ตัดวรรค หน้า-หลัง (Leading/Trailing spaces)
    df['organism_full'] = df['organism_full'].astype(str).str.strip()
    
    # 2. แก้ช่องว่างซ้อนกันตรงกลาง (Double spaces) ให้เหลือวรรคเดียว
    # เช่น 'E.  coli' -> 'E. coli'
    df['organism_full'] = df['organism_full'].str.replace(r'\s+', ' ', regex=True)

    # ==========================================
    # 3. เช็คหลังแก้ (After)
    # ==========================================
    new_names = df['organism_full'].unique()
    print(f"📊 จำนวนชื่อเชื้อหลังแก้: {len(new_names)} ชื่อ")
    
    if len(original_names) > len(new_names):
        print(f"✨ เยี่ยม! มีชื่อที่ซ้ำซ้อนกันหายไปจำนวน {len(original_names) - len(new_names)} ชื่อ")
    else:
        print("ℹ️ จำนวนชื่อเท่าเดิม (แสดงว่าชื่ออาจจะสะอาดอยู่แล้ว)")

    # ==========================================
    # 4. บันทึกไฟล์ (Save)
    # ==========================================
    print(f"💾 กำลังบันทึกทับไฟล์เดิม: {target_file}")
    df.to_csv(target_file, index=False)
    
    print("✅ เสร็จสมบูรณ์! แก้ไขเฉพาะชื่อเชื้อเรียบร้อยครับ")

if __name__ == "__main__":
    clean_names_no_calc()