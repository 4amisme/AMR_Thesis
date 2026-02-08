import pandas as pd
import os

def main():
    # --- 1. ตั้งค่า Path ---
    base_folder = os.path.join("MDR", "DrugClass2")
    
    # Input 1: ไฟล์สรุป MDR ล่าสุด (จากขั้นตอนที่แล้ว)
    input_summary = os.path.join(base_folder, "summary_MDR_only_by_year.csv")
    
    # Input 2: ไฟล์ ID
    input_id_file = os.path.join(base_folder, "ID.csv")
    
    # Output: ไฟล์ผลลัพธ์สุดท้าย
    output_filename = os.path.join(base_folder, "summary_MDR_final_with_ID.csv")

    # ตรวจสอบไฟล์
    print(f"Checking files in: {base_folder}")
    if not os.path.exists(input_summary):
        print(f"[Error] ไม่พบไฟล์สรุป: {input_summary}")
        print("กรุณารัน script สร้าง summary MDR ก่อนครับ")
        return
    if not os.path.exists(input_id_file):
        print(f"[Error] ไม่พบไฟล์ ID: {input_id_file}")
        return

    print("กำลังโหลดข้อมูล...")
    try:
        df_summary = pd.read_csv(input_summary, encoding='utf-8')
        df_id = pd.read_csv(input_id_file, encoding='utf-8')
    except Exception as e:
        print(f"[Error] อ่านไฟล์ CSV ไม่สำเร็จ: {e}")
        return

    # --- 2. Data Cleaning & Fixing Text ---
    
    # ลบช่องว่างชื่อคอลัมน์
    df_summary.columns = df_summary.columns.str.strip()
    df_id.columns = df_id.columns.str.strip()
    
    # *** แก้ไขคำผิด (Î² -> β) ในไฟล์ ID แบบอัตโนมัติ ***
    print("ตรวจสอบและแก้ไข Encoding (Î² -> β)...")
    wrong_text = "Î²-lactam combination agents"
    correct_text = "β-lactam combination agents"
    
    # เช็คและแก้ใน ID.csv (คอลัมน์ std_tested_list)
    if 'std_tested_list' in df_id.columns:
        df_id['std_tested_list'] = df_id['std_tested_list'].astype(str).str.replace(wrong_text, correct_text, regex=False)

    # --- 3. Normalization (จัดเรียงชื่อยาให้ตรงกัน) ---
    print("กำลังจัดรูปแบบข้อมูลเพื่อจับคู่ (Sort & Normalize)...")

    def normalize_drug_list(text):
        if pd.isna(text) or str(text).strip() == "" or str(text).lower() == "nan":
            return ""
        # 1. แยกด้วย comma
        parts = str(text).split(',')
        # 2. ตัดช่องว่างหน้าหลัง
        parts = [p.strip() for p in parts]
        # 3. เรียงลำดับ A-Z (สำคัญมาก! เพื่อให้ A,B ตรงกับ B,A)
        parts.sort()
        # 4. รวมกลับ
        return ", ".join(parts)

    # สร้างคอลัมน์ใหม่ชื่อ 'join_key' เพื่อใช้เชื่อมข้อมูล
    df_id['join_key'] = df_id['std_tested_list'].apply(normalize_drug_list)
    df_summary['join_key'] = df_summary['Resistant_Drug_Classes'].apply(normalize_drug_list)

    # --- 4. Merge Data (Mapping) ---
    print("กำลังจับคู่ข้อมูล MDR_id...")

    # Left Join: ยึดตาราง Summary เป็นหลัก, เอา MDR_id จาก ID มาแปะ
    merged_df = pd.merge(
        df_summary, 
        df_id[['join_key', 'MDR_id']], 
        on='join_key', 
        how='left'
    )

    # ลบคอลัมน์ช่วย (join_key) ทิ้ง
    merged_df.drop(columns=['join_key'], inplace=True)

    # --- 5. บันทึกผลลัพธ์ ---
    try:
        merged_df.to_csv(output_filename, index=False, encoding='utf-8')
        print("-" * 60)
        print(f"✅ เสร็จสมบูรณ์!")
        print(f"ไฟล์ผลลัพธ์สุดท้าย: {output_filename}")
        print("-" * 60)
        
        # แสดงตัวอย่าง
        print("\nตัวอย่างผลลัพธ์ (5 แถวแรกที่มี MDR_id):")
        # กรองให้เห็นเฉพาะตัวที่ map เจอ
        matched_rows = merged_df[merged_df['MDR_id'].notna()]
        
        if not matched_rows.empty:
            print(matched_rows[['x_year', 'Resistant_Drug_Classes', 'Count', 'MDR_id']].head())
        else:
            print(merged_df[['x_year', 'Resistant_Drug_Classes', 'Count', 'MDR_id']].head())
            print("\n[Note] ยังไม่พบรายการที่จับคู่เจอ (ลองเช็คชื่อยาใน ID.csv ว่าสะกดตรงกับ Summary หรือไม่)")

    except Exception as e:
        print(f"[Error] บันทึกไฟล์ไม่สำเร็จ: {e}")

if __name__ == "__main__":
    main()