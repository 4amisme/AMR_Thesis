import shutil
import os

# 1. กำหนดต้นทาง (Source) ตามที่คุณระบุมา
# ใส่ r ข้างหน้าเพื่อให้ Python อ่านเครื่องหมาย Backslash (\) ของ Windows ได้ถูกต้อง
source_path = r"C:\Users\ucwsh\Downloads\processed_2015.csv"

# 2. กำหนดปลายทาง (Destination)
# โฟลเดอร์ปลายทางในโปรเจกต์
target_folder = "data_for_mdr"
# ชื่อไฟล์ปลายทาง (ใช้ชื่อเดิมหรือเปลี่ยนก็ได้)
target_filename = "processed_2015.csv"

# รวม Path ปลายทางเข้าด้วยกัน
target_path = os.path.join(target_folder, target_filename)

def main():
    print(f"กำลังจะดึงไฟล์จาก: {source_path}")
    
    # ตรวจสอบว่ามีไฟล์ต้นทางอยู่จริงไหม
    if not os.path.exists(source_path):
        print("❌ ไม่เจอไฟล์ต้นทาง! กรุณาเช็คว่า path ถูกต้องหรือไม่")
        return

    # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    os.makedirs(target_folder, exist_ok=True)

    try:
        # สั่ง Copy ไฟล์ (ถ้าอยากย้ายเลย ให้เปลี่ยน .copy เป็น .move)
        shutil.copy(source_path, target_path)
        print("------------------------------------------------")
        print("✅ นำเข้าข้อมูลสำเร็จ!")
        print(f"ไฟล์ถูกบันทึกไว้ที่: {target_path}")
        print("------------------------------------------------")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()