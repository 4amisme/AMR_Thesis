import pandas as pd
import os
import glob

# 1. กำหนดรายการ Path ที่ต้องการเช็ค
paths = {
    "By Specimen": r"C:\AMR_Thesis\MDR\model_1Class\By specimen",
    "All Data": r"C:\AMR_Thesis\MDR\model_1Class\All Data",
    "By Ward Type": r"C:\AMR_Thesis\MDR\model_1Class\By ward type"
}

all_stats = []

print("⏳ กำลังเริ่มนับจำนวนแถวในแต่ละ Path...\n")

# 2. วนลูปไปในแต่ละ Path
for category, folder_path in paths.items():
    # ค้นหาไฟล์ CSV ทั้งหมด (รวมโฟลเดอร์ย่อย)
    files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    
    path_total_rows = 0
    file_count = 0
    
    print(f"📂 Category: {category}")
    print(f"{'File Name':<45} | {'Row Count':>10}")
    print("-" * 60)
    
    for f in files:
        try:
            # อ่านแค่คอลัมน์เดียวเพื่อประหยัด RAM ในการนับแถว
            df_temp = pd.read_csv(f, usecols=[0])
            row_count = len(df_temp)
            file_name = os.path.basename(f)
            
            print(f"{file_name[:45]:<45} | {row_count:>10,}")
            
            all_stats.append({
                'Category': category,
                'File_Name': file_name,
                'Row_Count': row_count
            })
            
            path_total_rows += row_count
            file_count += 1
        except Exception as e:
            print(f"❌ ไม่สามารถอ่านไฟล์ {os.path.basename(f)} ได้: {e}")
            
    print("-" * 60)
    print(f"📊 รวม {category}: {file_count} ไฟล์ | ยอดรวมทั้งหมด {path_total_rows:,} แถว\n")

# 3. บันทึกผลสรุปรวมเป็นไฟล์เดียว (แก้ไขจุดที่เกิด Error)
if len(all_stats) > 0:
    all_results = pd.DataFrame(all_stats)
    output_file = r"C:\AMR_Thesis\MDR\File_Row_Count_Summary.csv"
    all_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ ประมวลผลเสร็จสิ้น! บันทึกตารางสรุปไว้ที่: {output_file}")
else:
    print("❌ ไม่พบข้อมูลไฟล์ CSV ใน Path ที่กำหนด โปรดตรวจสอบความถูกต้องของ Path ครับ")