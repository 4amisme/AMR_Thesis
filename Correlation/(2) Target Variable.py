import pandas as pd
import numpy as np

# 1. กำหนด Path ของไฟล์ (สามารถใช้ Path ที่ให้มาได้เลย)
file_path = 'Correlation/AllYears_Resistance_Full_Analysis.csv'

# 2. โหลดไฟล์ข้อมูล
# กำหนด encoding='utf-8' เพื่อป้องกันปัญหาการอ่านตัวอักษรหรือชื่อเชื้อแปลกๆ
try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print("✅ โหลดไฟล์ข้อมูลสำเร็จ")
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ที่ระบุ กรุณาตรวจสอบ Path: {file_path}")

# 3. ตรวจสอบชื่อคอลัมน์ให้แน่ใจ (ระวังเรื่องตัวพิมพ์เล็ก-ใหญ่ เช่น class_count vs Class_Count)
# หากในไฟล์จริงเป็นตัวพิมพ์ใหญ่ สามารถแก้โค้ดด้านล่างให้ตรงกันได้เลย
target_col = 'class_count'

if target_col in df.columns:
    # 4. สร้างคอลัมน์ MDR_status 
    # ใช้ np.where: ถ้าน้อยกว่า 3 เป็น 0 (Non-MDR), ถ้าตั้งแต่ 3 ขึ้นไปเป็น 1 (MDR)
    df['MDR_status'] = np.where(df[target_col] >= 3, 1, 0)
    
    # --- หรือสามารถใช้วิธีที่สั้นกว่าได้เช่นกัน ---
    # df['MDR_status'] = (df[target_col] >= 3).astype(int)

    print("\n[Preview] ตัวอย่างข้อมูล 5 แถวแรกหลังเพิ่มคอลัมน์ MDR_status:")
    print(df[[target_col, 'MDR_status']].head())
    
    # 5. บันทึกไฟล์ข้อมูลชุดใหม่เพื่อนำไปใช้วิเคราะห์ต่อ (Optional)
    output_path = 'Correlation/MDR_for_correlation.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ บันทึกไฟล์ข้อมูลพร้อมสถานะ MDR แล้วที่: {output_path}")

else:
    print(f"❌ ไม่พบคอลัมน์ '{target_col}' ในข้อมูล")
    print(f"คอลัมน์ที่มีทั้งหมดคือ: {list(df.columns)}")