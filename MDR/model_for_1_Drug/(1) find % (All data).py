import pandas as pd
import os

def main():
    # --- 1. CONFIGURATION ---
    input_path = r"C:\AMR_Thesis\MDR\model_for_1_Drug\AllYears_DrugClass_Exploded.csv"
    output_dir = r"C:\AMR_Thesis\MDR\model_for_1_Drug\All_data_including_S" # เปลี่ยนชื่อโฟลเดอร์ให้สื่อความหมาย
    
    target_organisms = [
        "Acinetobacter baumannii", "Enterococcus faecalis", "Enterococcus faecium",
        "Escherichia coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa", "Staphylococcus aureus"
    ]
    target_lower = [org.lower().strip() for org in target_organisms]

    os.makedirs(output_dir, exist_ok=True)
    print("🚀 เริ่มต้นการประมวลผล (นับรวมเคสที่ไม่ดื้อยาด้วย)...")

    temp_storage = {org: [] for org in target_organisms}

    try:
        # --- 2. LOAD DATA IN CHUNKS ---
        chunk_size = 100000 
        reader = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)

        chunk_count = 0
        for chunk in reader:
            chunk_count += 1
            chunk.columns = chunk.columns.str.strip().str.lower()
            
            # กรอง Missing_Count (ถ้ามี)
            if 'missing_count' in chunk.columns:
                chunk = chunk[chunk['missing_count'] == 0]
            
            # --- ส่วนที่แก้ไข: นำบรรทัด dropna(subset=['resistant_drug_name']) ออกแล้ว ---
            # เราจะเก็บทุกแถวไว้ แม้ว่า resistant_drug_name จะเป็น NaN (แปลว่าเชื้อไวต่อยา/ไม่ดื้อ)
            
            # จัดการเรื่องวันที่
            chunk['spec_date'] = pd.to_datetime(chunk['spec_date'], errors='coerce')
            chunk = chunk.dropna(subset=['spec_date'])
            chunk = chunk[(chunk['spec_date'].dt.year >= 2015) & (chunk['spec_date'].dt.year <= 2024)]
            
            if chunk.empty:
                continue

            chunk['year'] = chunk['spec_date'].dt.year.astype(int)
            chunk['month'] = chunk['spec_date'].dt.month.astype(int)
            chunk['org_clean'] = chunk['organism_full'].str.lower().str.strip()

            # แยกข้อมูลลงถัง
            for org_name, org_lower in zip(target_organisms, target_lower):
                match_data = chunk[chunk['org_clean'] == org_lower].copy()
                if not match_data.empty:
                    temp_storage[org_name].append(match_data)
            
            if chunk_count % 10 == 0:
                print(f"⏳ ประมวลผลไปแล้ว {chunk_count * chunk_size:,} แถว...")

        # --- 3. AGGREGATE AND SAVE ---
        print("\n📊 กำลังสรุปผลรวมสัดส่วน %...")
        
        for organism, data_list in temp_storage.items():
            if not data_list:
                print(f"⚠️ ไม่พบข้อมูลสำหรับเชื้อ: {organism}")
                continue
            
            df_org = pd.concat(data_list, ignore_index=True)
            
            # คำนวณตัวหาร: จำนวน records ทั้งหมดของเชื้อนี้ (รวมทั้งดื้อและไม่ดื้อ)
            monthly_total = df_org.groupby(['year', 'month']).size().reset_index(name='total_records_in_month')
            
            # คำนวณตัวตั้ง: แยกตามชื่อยา (แถวที่เป็น NaN จะถูกนับรวมในกลุ่มยาว่าง)
            monthly_counts = df_org.groupby(['year', 'month', 'resistant_drug_name'], dropna=False).size().reset_index(name='drug_count')
            
            final_df = pd.merge(monthly_counts, monthly_total, on=['year', 'month'])
            final_df['percentage'] = ((final_df['drug_count'] / final_df['total_records_in_month']) * 100).round(2)
            final_df['organism_full'] = organism

            # เรียงลำดับ (เอาดื้อมากไว้บน เคสที่ไม่ดื้อจะอยู่ล่างๆ)
            final_df = final_df.sort_values(['year', 'month', 'percentage'], ascending=[True, True, False])
            
            safe_name = organism.replace(" ", "_").replace(".", "") + ".csv"
            final_df.to_csv(os.path.join(output_dir, safe_name), index=False, encoding='utf-8-sig')
            print(f"💾 บันทึกสำเร็จ: {safe_name}")

        print(f"\n✨ ดำเนินการเสร็จสมบูรณ์! ผลลัพธ์อยู่ที่: {output_dir}")

    except Exception as e:
        print(f"🔴 เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()