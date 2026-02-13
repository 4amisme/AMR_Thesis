import pandas as pd
import os

# 1. กำหนด Path
data_path = os.path.join("MDR", "data", "AllYears_processed.csv")
mapping_path = os.path.join("MDR", "DrugClass1.2", "Drug_class_for_MDR_new.csv")
output_path = os.path.join("MDR", "DrugClass1.2", "AllYears_DrugClass_tested.csv")

# 2. โหลดข้อมูล
df_data = pd.read_csv(data_path)
df_mapping = pd.read_csv(mapping_path)

# 3. จัดการข้อมูลมาตรฐาน (Drug Class Mapping)
# ทำการ Strip ช่องว่าง และแปลงชื่อ Antibiotic เป็นตัวพิมพ์ใหญ่ทั้งหมดเพื่อใช้เป็น Key
df_mapping['Antibiotic_Std'] = df_mapping['Antibiotic'].astype(str).str.strip().str.upper()
df_mapping['Class_Std'] = df_mapping['Class'].astype(str).str.strip()

# สร้าง Dictionary สำหรับ Mapping และ Set ของ Class ทั้งหมด
mapping_dict = dict(zip(df_mapping['Antibiotic_Std'], df_mapping['Class_Std']))
all_standard_classes = set(df_mapping['Class_Std'].unique())

# 4. ตรวจสอบชื่อคอลัมน์ในไฟล์ข้อมูล
# สร้าง Mapping ระหว่าง "ชื่อคอลัมน์เดิม" กับ "ชื่อที่ทำ Standardize แล้ว"
# เพื่อให้เรายังเข้าถึงคอลัมน์เดิมได้ แม้ชื่อในไฟล์จะสะกดด้วยตัวพิมพ์เล็กหรือมีช่องว่าง
col_to_std = {col: str(col).strip().upper() for col in df_data.columns}

# กรองเฉพาะคอลัมน์ที่มีชื่อตรงกับในตารางยามาตรฐาน
# valid_cols_map จะเก็บ { 'ชื่อคอลัมน์เดิม': 'ชื่อมาตรฐาน(ตัวใหญ่)' }
valid_cols_map = {orig: std for orig, std in col_to_std.items() if std in mapping_dict}

print(f"พบยาที่ตรงตามมาตรฐานทั้งหมด {len(valid_cols_map)} คอลัมน์")

# 5. ฟังก์ชันวิเคราะห์ข้อมูลแต่ละแถว
def analyze_drug_classes_robust(row):
    tested_classes = set()
    
    # วนลูปเฉพาะคอลัมน์ยาที่ตรวจพบว่าตรงกับมาตรฐาน
    for orig_col, std_name in valid_cols_map.items():
        val = row[orig_col]
        # ตรวจสอบค่าในช่อง: ต้องไม่เป็น NaN และเมื่อตัดช่องว่างแล้วต้องไม่เป็นค่าว่าง
        if pd.notna(val) and str(val).strip() != "":
            # ดึงชื่อ Class จาก Dictionary โดยใช้ชื่อมาตรฐาน (ตัวพิมพ์ใหญ่)
            drug_class = mapping_dict[std_name]
            tested_classes.add(drug_class)
    
    # คำนวณส่วนต่างของ Class
    missing_classes = all_standard_classes - tested_classes
    
    return pd.Series({
        'Tested_Classes': ", ".join(sorted(list(tested_classes))),
        'Missing_Classes': ", ".join(sorted(list(missing_classes))),
        'Missing_Count': len(missing_classes)
    })

# 6. ประมวลผลและรวมข้อมูล (Keep original columns intact)
analysis_results = df_data.apply(analyze_drug_classes_robust, axis=1)
df_final = pd.concat([df_data, analysis_results], axis=1)

# 7. บันทึกไฟล์
df_final.to_csv(output_path, index=False)

print(f"ประมวลผลเสร็จสิ้น! บันทึกไฟล์ที่: {output_path}")