import pandas as pd
import os

def load_data():
    # ==========================================
    # ตั้งค่า Path ต้นทาง (Source)
    # ==========================================
    source_path = r"C:\Users\ucwsh\Downloads\DrugClass - New.csv"
    
    # ไฟล์ปลายทาง (Intermediate file) ที่จะส่งต่อให้ขั้นตอนถัดไป
    destination_file = os.path.join("MDR", "DrugClass1.2", "Drug_class_for_MDR_new.csv")

    print(f"[Step 1] กำลังอ่านข้อมูลจาก: {source_path}")

    # 1. เช็คว่าเจอไฟล์ไหม
    if not os.path.exists(source_path):
        print("Error: หาไฟล์ต้นทางไม่เจอ กรุณาตรวจสอบ Path")
        return

    # 2. อ่านไฟล์
    try:
        df = pd.read_csv(source_path)
        print(f"อ่านไฟล์สำเร็จ! พบข้อมูล {len(df)} แถว")
    except Exception as e:
        print(f"Error ในการอ่าน CSV: {e}")
        return

    # 3. บันทึกไฟล์พักไว้ในโฟลเดอร์ปัจจุบัน (เพื่อส่งต่อให้ Step 2)
    # (เราบันทึกทับไฟล์เดิมได้เลย เพื่อประหยัดที่)
    df.to_csv(destination_file, index=False)
    
    print(f"[Step 1] บันทึกข้อมูลสำหรับใช้งานต่อที่: {destination_file}")
    print("--- จบขั้นตอนโหลดข้อมูล ---")

if __name__ == "__main__":
    load_data()