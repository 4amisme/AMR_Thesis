import os
import pandas as pd

# 1. กำหนด Path
base_path = r"C:\Users\MP1YXGGZ\Documents\GitHub\AMR_Thesis\MDR\model"
sub_folders = ["All Data", "By ward type", "By_specimen"]
output_file = os.path.join(base_path, "drug_resistance_patterns_summary.csv")

def process_resistance_patterns():
    all_summaries = []
    
    species_col = 'organism_full'
    target_col = 'Resistant_Drug_Classes'
    
    for folder in sub_folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path): continue
            
        print(f"Analyzing resistance patterns in: {folder}...")
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    
                    if target_col in df.columns and species_col in df.columns:
                        # 1. จัดการข้อมูลเบื้องต้น (ลบค่าว่างและตัดช่องว่างหัวท้าย)
                        df_temp = df[[species_col, target_col]].copy()
                        df_temp[target_col] = df_temp[target_col].fillna('No Resistance Identified').str.strip()
                        
                        # 2. นับจำนวนตาม "กลุ่มยาเดิม" ที่มาในไฟล์ (ไม่ระเบิดแถว)
                        summary = df_temp.groupby([species_col, target_col]).size().reset_index(name='Count')
                        
                        # 3. คำนวณเปอร์เซ็นต์เทียบกับเชื้อชนิดนั้นทั้งหมดในไฟล์
                        total_per_species = df[species_col].value_counts().to_dict()
                        summary['Total_Species_In_File'] = summary[species_col].map(total_per_species)
                        summary['Percentage'] = (summary['Count'] / summary['Total_Species_In_File'] * 100).round(2)
                        
                        # เพิ่มแหล่งที่มา
                        summary['Source_Folder'] = folder
                        summary['Source_File'] = filename
                        
                        all_summaries.append(summary)
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # 2. รวมผลลัพธ์และบันทึก
    if all_summaries:
        final_df = pd.concat(all_summaries, ignore_index=True)
        
        # จัดเรียงคอลัมน์และลำดับข้อมูล (เรียงจากจำนวนมากไปน้อยในแต่ละเชื้อ)
        final_df = final_df.sort_values(['Source_File', species_col, 'Count'], ascending=[True, True, False])
        
        cols = ['Source_Folder', 'Source_File', species_col, target_col, 'Count', 'Total_Species_In_File', 'Percentage']
        final_df = final_df[cols]
        
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n--- วิเคราะห์รูปแบบการดื้อยาสำเร็จ! ---")
        print(f"บันทึกไฟล์ไว้ที่: {output_file}")
    else:
        print("ไม่พบข้อมูลที่ต้องการวิเคราะห์")

if __name__ == "__main__":
    process_resistance_patterns()