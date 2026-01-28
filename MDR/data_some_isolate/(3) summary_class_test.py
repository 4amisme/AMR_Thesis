import pandas as pd
import os

def summarize_tested_patterns():
    # ==========================================
    # 1. ตั้งค่าไฟล์
    # ==========================================
    # อ่านจากไฟล์ที่ได้จาก Step ก่อนหน้า
    input_file = os.path.join("MDR", "data_some_isolate", "AllYears_Checked_Classes.csv")
    output_file = os.path.join("MDR", "data_some_isolate", "Tested_Classes_Summary.csv")

    print(f"🚀 กำลังสรุปรูปแบบการตรวจยา (Grouping Patterns)...")
    print(f"📂 อ่านไฟล์: {input_file}")

    if not os.path.exists(input_file):
        print(f"❌ Error: ไม่พบไฟล์ {input_file}")
        return

    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่สำเร็จ: {e}")
        return

    # ==========================================
    # 2. จำลองคำสั่ง SQL
    # ==========================================
    # SQL:
    # SELECT list_tested_classes, count_tested_classes
    # FROM AllYears_Checked_Classes
    # GROUP BY list_tested_classes, count_tested_classes
    # ORDER BY count_tested_classes
    
    print("📊 กำลังประมวลผล Group By...")

    # ใช้ groupby().size() เพื่อยุบรวมแถวที่เหมือนกัน และนับจำนวน
    # (Reset index เพื่อให้กลับมาเป็นตารางปกติ)
    summary_df = df.groupby(['list_tested_classes', 'count_tested_classes']) \
                   .size() \
                   .reset_index(name='isolate_count') # ตั้งชื่อคอลัมน์นับจำนวนว่า isolate_count

    # ORDER BY count_tested_classes
    # (เรียงตามจำนวน Class น้อย->มาก และถ้าเท่ากัน ให้เรียงตามจำนวนคนที่พบ มาก->น้อย จะได้ดูง่าย)
    summary_df = summary_df.sort_values(by=['count_tested_classes', 'isolate_count'], ascending=[True, False])

    # ==========================================
    # 3. บันทึกผลลัพธ์
    # ==========================================
    print(f"💾 กำลังบันทึกไฟล์สรุป: {output_file}")
    summary_df.to_csv(output_file, index=False)
    
    print("-" * 60)
    print("✅ เสร็จสมบูรณ์! ตัวอย่างผลลัพธ์ (Top 10 Patterns):")
    print(summary_df.head(10))
    print("-" * 60)
    print(f"💡 ไฟล์นี้จะบอกคุณว่า 'การตรวจรูปแบบไหน' ที่เจอบ่อยที่สุด")
    print("   เช่น ตรวจครบ 3 ตัว (Aminoglycosides, Carbapenems, Cephalosporins) มีกี่เคส")

if __name__ == "__main__":
    summarize_tested_patterns()