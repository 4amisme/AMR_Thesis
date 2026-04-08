import pandas as pd
import os

def main():
    # --- 1. CONFIG PATH ---
    # ตรวจสอบให้แน่ใจว่าชื่อไฟล์ Input ตรงกับที่คุณบันทึกไว้จากขั้นตอนที่แล้ว
    input_path = os.path.join("MDR", "DrugClass2", "AllYears_SingleClass_Pattern.csv")
    output_dir = os.path.join("MDR", "model")

    # สร้างโฟลเดอร์รองรับไฟล์ผลลัพธ์
    os.makedirs(output_dir, exist_ok=True)

    print("--- เริ่มต้นการประมวลผลข้อมูลรายเดือน (ฉบับสมบูรณ์) ---")

    try:
        # --- 2. LOAD DATA ---
        # ใช้ low_memory=False เพื่อแก้ปัญหา DtypeWarning ที่คุณพบ
        df = pd.read_csv(input_path, low_memory=False)
        print(f"✅ โหลดไฟล์สำเร็จ: {len(df)} แถว")

        # --- 3. STANDARDIZE COLUMNS ---
        # ปรับหัวตารางเป็นตัวเล็กทั้งหมดเพื่อป้องกันปัญหา Error: 'Resistant_Drug_Classes'
        df.columns = df.columns.str.strip().str.lower()

        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_cols = ['spec_date', 'organism_full', 'resistant_drug_classes']
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ ไม่พบคอลัมน์ที่จำเป็น: '{col}'")
                return

        # --- 4. DATE PREPARATION ---
        # แปลงวันที่และกรองช่วงปี 2015-2024
        df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
        df = df.dropna(subset=['spec_date'])
        
        df = df[(df['spec_date'].dt.year >= 2015) & (df['spec_date'].dt.year <= 2024)].copy()
        
        df['year'] = df['spec_date'].dt.year.astype(int)
        df['month'] = df['spec_date'].dt.month.astype(int)

        # --- 5. PROCESS BY ORGANISM ---
        all_organisms = df['organism_full'].unique()
        print(f"🧬 กำลังประมวลผลเชื้อทั้งหมด {len(all_organisms)} ชนิด...")

        for organism in all_organisms:
            if pd.isna(organism): continue
            
            # กรองข้อมูลเฉพาะเชื้อตัวนั้นๆ
            df_org = df[df['organism_full'] == organism].copy()
            
            # กรมข้อมูลตัวหาร (จำนวนเคสรวมของเชื้อนี้ในแต่ละเดือน)
            monthly_total = df_org.groupby(['year', 'month']).size().reset_index(name='total_rows_in_month')

            # นับจำนวนการเกิดของ "ทุก Pattern" (ไม่มีการตัด Top 5 ออก)
            monthly_counts = df_org.groupby(['year', 'month', 'resistant_drug_classes']).size().reset_index(name='pattern_count')

            # รวมข้อมูลตัวตั้งและตัวหารเข้าด้วยกัน
            final_df = pd.merge(monthly_counts, monthly_total, on=['year', 'month'])
            
            # คำนวณเปอร์เซ็นต์
            final_df['percentage'] = ((final_df['pattern_count'] / final_df['total_rows_in_month']) * 100).round(2)
            final_df['organism_full'] = organism

            # จัดลำดับคอลัมน์ให้สวยงาม
            column_order = [
                'organism_full', 
                'year', 
                'month', 
                'resistant_drug_classes', 
                'pattern_count', 
                'total_rows_in_month', 
                'percentage'
            ]
            final_df = final_df[column_order]
            
            # เรียงลำดับตาม เวลา และ เปอร์เซ็นต์จากมากไปน้อย
            final_df = final_df.sort_values(['year', 'month', 'percentage'], ascending=[True, True, False])

            # --- 6. SAVE INDIVIDUAL CSV ---
            # ล้างชื่อเชื้อให้เป็นชื่อไฟล์ที่ปลอดภัย (ไม่มีอักขระพิเศษ)
            clean_name = "".join([c if c.isalnum() else "_" for c in str(organism).lower()])
            file_name = f"{clean_name}.csv"
            
            output_file_path = os.path.join(output_dir, file_name)
            final_df.to_csv(output_file_path, index=False, encoding='utf-8')
            
        print(f"\n✨ ดำเนินการเสร็จสมบูรณ์!")
        print(f"📂 ผลลัพธ์แยกรายไฟล์อยู่ในโฟลเดอร์: {output_dir}")

    except Exception as e:
        print(f"🔴 เกิดข้อผิดพลาดระหว่างประมวลผล: {e}")

if __name__ == "__main__":
    main()