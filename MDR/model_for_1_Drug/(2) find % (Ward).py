import pandas as pd
import os

def main():
    # --- 1. CONFIGURATION ---
    input_path = r"C:\AMR_Thesis\MDR\model_for_1_Drug\AllYears_DrugClass_Exploded.csv"
    base_output_dir = r"C:\AMR_Thesis\MDR\model_for_1_Drug"
    
    target_organisms = [
        "Acinetobacter baumannii", "Enterococcus faecalis", "Enterococcus faecium",
        "Escherichia coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa", "Staphylococcus aureus"
    ]
    target_lower = [org.lower().strip() for org in target_organisms]

    os.makedirs(base_output_dir, exist_ok=True)
    print("🚀 เริ่มต้นการประมวลผล (นับรวมเคสที่ไม่ดื้อยาเข้าในสถิติ)...")

    temp_storage = {}

    try:
        # --- 2. LOAD DATA IN CHUNKS ---
        chunk_size = 200000 
        reader = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)

        for i, chunk in enumerate(reader):
            chunk.columns = chunk.columns.str.strip().str.lower()
            
            # --- 3. CLEANING & STANDARDIZING ---
            # แก้ไข: นำ 'resistant_drug_name' ออกจาก dropna เพื่อรักษาเคสที่ไม่ดื้อยาไว้
            chunk = chunk.dropna(subset=['ward_type', 'spec_date', 'organism_full'])

            chunk['ward_type'] = chunk['ward_type'].astype(str).str.strip().str.lower()
            chunk['ward_type'] = chunk['ward_type'].replace({'ccu': 'icu'})
            chunk = chunk[chunk['ward_type'].isin(['icu', 'in', 'out'])]

            chunk['org_lower'] = chunk['organism_full'].astype(str).str.strip().str.lower()
            chunk = chunk[chunk['org_lower'].isin(target_lower)]

            chunk['spec_date'] = pd.to_datetime(chunk['spec_date'], errors='coerce')
            chunk = chunk.dropna(subset=['spec_date'])
            chunk = chunk[(chunk['spec_date'].dt.year >= 2015) & (chunk['spec_date'].dt.year <= 2024)]
            
            if chunk.empty: continue

            chunk['year'] = chunk['spec_date'].dt.year.astype(int)
            chunk['month'] = chunk['spec_date'].dt.month.astype(int)

            for organism_lower in chunk['org_lower'].unique():
                original_name = next(org for org in target_organisms if org.lower() == organism_lower)
                
                if original_name not in temp_storage:
                    temp_storage[original_name] = []
                temp_storage[original_name].append(chunk[chunk['org_lower'] == organism_lower].copy())

            if (i + 1) % 5 == 0:
                print(f"⏳ อ่านไฟล์ไปแล้ว {(i + 1) * chunk_size:,} แถว...")

        # --- 4. AGGREGATE & SAVE ---
        print("\n📊 กำลังสรุปผลแยกตาม Ward และบันทึกไฟล์ (รวมเคส Sensitive)...")
        
        for organism, data_list in temp_storage.items():
            df_org = pd.concat(data_list, ignore_index=True)
            
            clean_org_folder = "".join([c if c.isalnum() else "_" for c in organism.lower()])
            org_dir = os.path.join(base_output_dir, clean_org_folder)
            os.makedirs(org_dir, exist_ok=True)

            # ตัวหาร: จำนวนแถวทั้งหมด (รวมเคสที่ยาเป็นค่าว่าง)
            monthly_ward_total = df_org.groupby(['year', 'month', 'ward_type']).size().reset_index(name='total_rows_in_ward_month')
            
            # ตัวตั้ง: นับจำนวนยา (ใช้ dropna=False เพื่อให้นับกลุ่มที่ 'ไม่ดื้อยา' ออกมาเป็นอีกหนึ่งแถว)
            monthly_ward_counts = df_org.groupby(['year', 'month', 'ward_type', 'resistant_drug_name'], dropna=False).size().reset_index(name='drug_count')

            final_df = pd.merge(monthly_ward_counts, monthly_ward_total, on=['year', 'month', 'ward_type'])
            final_df['percentage'] = ((final_df['drug_count'] / final_df['total_rows_in_ward_month']) * 100).round(2)
            final_df['organism_full'] = organism

            for ward in ['icu', 'in', 'out']:
                ward_df = final_df[final_df['ward_type'] == ward].copy()
                if ward_df.empty: continue
                
                ward_df = ward_df.sort_values(['year', 'month', 'percentage'], ascending=[True, True, False])
                
                parts = organism.lower().split()
                short_name = f"{parts[0][0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
                file_name = f"{short_name}_{ward}.csv"
                
                ward_df.to_csv(os.path.join(org_dir, file_name), index=False, encoding='utf-8-sig')

        print(f"\n✨ เสร็จสมบูรณ์! ไฟล์ในโฟลเดอร์เหล่านี้จะรวมสถิติเคสที่ไม่ดื้อยาด้วย")

    except Exception as e:
        print(f"🔴 เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()