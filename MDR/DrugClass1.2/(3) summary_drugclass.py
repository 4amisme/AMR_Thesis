import pandas as pd
import os

def main():
    # --- ส่วนที่ 1: กำหนด Path ---
    # อ้างอิงไฟล์จากผลลัพธ์ของขั้นตอนที่แล้ว
    input_path = os.path.join("MDR", "DrugClass1.2", "AllYears_DrugClass_tested.csv")
    output_dir = os.path.join("MDR", "DrugClass1.2")
    summary_output_path = os.path.join(output_dir, "Summary_DrugClass_Testing.csv")

    if not os.path.exists(input_path):
        print(f"❌ ไม่พบไฟล์ผลลัพธ์: {input_path}")
        print("กรุณารันสคริปต์หลักเพื่อสร้างไฟล์ Filtered ก่อนครับ")
        return

    print("🚀 เริ่มต้นการทำ Summary ข้อมูล...")

    # --- ส่วนที่ 2: โหลดข้อมูล ---
    # ใช้ low_memory=False เนื่องจากไฟล์หลักน่าจะใหญ่
    df = pd.read_csv(input_path, low_memory=False)

    # รายชื่อคอลัมน์ที่ต้องการนำมา Group
    group_cols = ['ORGANISM_FULL', 'Tested_Classes', 'Missing_Classes', 'Missing_Count']

    # ตรวจสอบว่ามีคอลัมน์ครบถ้วนหรือไม่
    missing_from_df = [col for col in group_cols if col not in df.columns]
    if missing_from_df:
        print(f"❌ ไม่พบคอลัมน์ต่อไปนี้ในไฟล์: {missing_from_df}")
        return

    # --- ส่วนที่ 3: การ GroupBy และนับจำนวน ---
    # 1. จัดการค่าว่างในคอลัมน์ที่จะ Group (เพื่อไม่ให้ถูกคัดออกตอนนับ)
    df_temp = df[group_cols].copy()
    df_temp['Missing_Classes'] = df_temp['Missing_Classes'].fillna('None (Fully Tested)')

    # 2. ทำการ GroupBy และนับจำนวน Record (Count)
    summary_df = df_temp.groupby(group_cols, as_index=False).size()

    # 3. เปลี่ยนชื่อคอลัมน์จาก size เป็น Count เพื่อความเข้าใจง่าย
    summary_df = summary_df.rename(columns={'size': 'Record_Count'})

    # 4. เรียงลำดับข้อมูล (เรียงตามชื่อเชื้อ และจำนวนจากมากไปน้อย)
    summary_df = summary_df.sort_values(by=['ORGANISM_FULL', 'Record_Count'], ascending=[True, False])

    # --- ส่วนที่ 4: บันทึกไฟล์ ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    summary_df.to_csv(summary_output_path, index=False, encoding='utf-8')
    
    print("-" * 30)
    print(f"🎉 สร้างไฟล์ Summary สำเร็จ!")
    print(f"📁 บันทึกที่: {summary_output_path}")
    print(f"📊 จำนวนรูปแบบการตรวจที่พบ: {len(summary_df)} รูปแบบ")
    print("-" * 30)
    
    # แสดงตัวอย่างหน้าจอ
    print("\nตัวอย่างข้อมูล Summary (5 แถวแรก):")
    print(summary_df.head())

if __name__ == "__main__":
    main()