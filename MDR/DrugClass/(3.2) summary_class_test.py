import pandas as pd
import os

def summarize_tested_patterns_by_organism():
    # ==========================================
    # 1. ตั้งค่าไฟล์
    # ==========================================
    input_file = os.path.join("MDR", "DrugClass", "AllYears_Checked_Classes.csv")
    output_file = os.path.join("MDR", "DrugClass", "Tested_Classes_Summary_By_Organism.csv")

    print(f"🚀 กำลังสรุปรูปแบบการตรวจยาแยกตามเชื้อ (Grouping Patterns by Organism)...")
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
    # 2. จัดการชื่อคอลัมน์ให้ตรงกับโจทย์
    # ==========================================
    # ทำชื่อคอลัมน์ให้เป็นตัวเล็กและตัดช่องว่าง
    df.columns = df.columns.str.strip().str.lower()

    # เช็คว่ามีคอลัมน์ organism หรือ organism_full หรือไม่
    # ถ้ามี 'organism' แต่ไม่มี 'organism_full' ให้เปลี่ยนชื่อเพื่อให้โค้ดทำงานต่อได้
    if 'organism' in df.columns and 'organism_full' not in df.columns:
        df.rename(columns={'organism': 'organism_full'}, inplace=True)
    
    if 'organism_full' not in df.columns:
        print("❌ Error: ไม่พบคอลัมน์ชื่อเชื้อ (organism หรือ organism_full)")
        return

    # ==========================================
    # 3. Group By: Organism + Pattern
    # ==========================================
    # SQL Logic:
    # SELECT organism_full, list_tested_classes, count_tested_classes, COUNT(*)
    # GROUP BY organism_full, list_tested_classes, count_tested_classes
    
    print("📊 กำลังประมวลผล Group By...")

    summary_df = df.groupby(['organism_full', 'list_tested_classes', 'count_tested_classes']) \
                   .size() \
                   .reset_index(name='isolate_count')

    # ==========================================
    # 4. เรียงลำดับ (Sorting)
    # ==========================================
    # เรียงตาม:
    # 1. ชื่อเชื้อ (A -> Z)
    # 2. จำนวน Class ที่ตรวจ (น้อย -> มาก)
    # 3. จำนวนเคสที่เจอ (มาก -> น้อย)
    summary_df = summary_df.sort_values(
        by=['organism_full', 'count_tested_classes', 'isolate_count'], 
        ascending=[True, True, False]
    )

    # ==========================================
    # 5. บันทึกผลลัพธ์
    # ==========================================
    print(f"💾 กำลังบันทึกไฟล์สรุป: {output_file}")
    summary_df.to_csv(output_file, index=False)
    
    print("-" * 80)
    print("✅ เสร็จสมบูรณ์! ตัวอย่างผลลัพธ์ (แสดงบางส่วน):")
    # แสดงตัวอย่างโดยเลือกมาสัก 2 เชื้อ เพื่อให้เห็นภาพ
    print(summary_df.head(10))
    print("-" * 80)

if __name__ == "__main__":
    summarize_tested_patterns_by_organism()