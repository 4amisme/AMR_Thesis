import pandas as pd
import os

def aggregate_data_for_model():
    # ==========================================
    # 1. ตั้งค่าไฟล์
    # ==========================================
    input_file = os.path.join("MDR", "data_for_model", "AllYears_MDR_Final.csv")
    output_file = os.path.join("MDR", "data_for_model", "AllYears_MDR_for_model.csv")

    print(f"🚀 เริ่มกระบวนการเตรียมข้อมูลสำหรับ Model...")
    print(f"📂 กำลังอ่านไฟล์: {input_file}")

    if not os.path.exists(input_file):
        print(f"❌ Error: ไม่พบไฟล์ {input_file}")
        return

    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่สำเร็จ: {e}")
        return

    # ==========================================
    # 2. เตรียมข้อมูล (Pre-processing)
    # ==========================================
    print("🧹 กำลังจัดการวันที่และคอลัมน์...")
    
    if 'organism' in df.columns and 'organism_full' not in df.columns:
        df.rename(columns={'organism': 'organism_full'}, inplace=True)
    
    df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
    df['year_val'] = df['spec_date'].dt.strftime('%Y')
    df['month_val'] = df['spec_date'].dt.strftime('%m')

    # ==========================================
    # 3. กรองข้อมูล (WHERE Clause)
    # ==========================================
    print("🔍 กำลังกรองข้อมูล...")
    df_clean = df.dropna(subset=['organism_full', 'drug_classes', 'spec_date'])
    df_clean = df_clean[df_clean['year_val'].between('2015', '2024')]

    # ==========================================
    # 4. คำนวณยอดรวม (Aggregation)
    # ==========================================
    print("∑ กำลังคำนวณยอดรวม (Group By)...")

    # สร้างคอลัมน์ตัวเลข 0/1 เพื่อเตรียม Sum
    df_clean['is_R'] = (df_clean['MDR'] == 'R').astype(int)
    df_clean['is_S'] = df_clean['MDR'].isin(['S']).astype(int)
    df_clean['is_Total'] = df_clean['MDR'].isin(['R', 'S']).astype(int)

    # Group By และ Sum
    grouped_df = df_clean.groupby(['organism_full', 'year_val', 'month_val'])[[
        'is_R', 'is_S', 'is_Total'
    ]].sum().reset_index()

    # เปลี่ยนชื่อคอลัมน์ให้สื่อความหมาย
    grouped_df.rename(columns={
        'is_R': 'count_R',
        'is_S': 'count_S',
        'is_Total': 'total_count'
    }, inplace=True)

    # ==========================================
    # 5. [ใหม่!] คำนวณ %R (Percentage)
    # ==========================================
    print("🧮 กำลังคำนวณ %R...")
    
    # สูตร: (count_R / total_count) * 100
    # ใช้ round(2) เพื่อปัดทศนิยม 2 ตำแหน่ง
    grouped_df['percent_R'] = (grouped_df['count_R'] / grouped_df['total_count']) * 100
    grouped_df['percent_R'] = grouped_df['percent_R'].round(2)
    
    # (ป้องกัน Error กรณี total_count เป็น 0 แม้โอกาสน้อย แต่ใส่ fillna(0) ไว้เพื่อความชัวร์)
    grouped_df['percent_R'] = grouped_df['percent_R'].fillna(0)

    # จัดเรียงข้อมูล
    grouped_df = grouped_df.sort_values(by=['organism_full', 'year_val', 'month_val'])

    # ==========================================
    # 6. บันทึกไฟล์
    # ==========================================
    print(f"💾 กำลังบันทึกไฟล์ผลลัพธ์: {output_file}")
    grouped_df.to_csv(output_file, index=False)
    
    print("-" * 60)
    print("✅ เสร็จสมบูรณ์! ตัวอย่างข้อมูล (พร้อมคอลัมน์ percent_R):")
    print(grouped_df.head(10))
    print("-" * 60)

if __name__ == "__main__":
    aggregate_data_for_model()