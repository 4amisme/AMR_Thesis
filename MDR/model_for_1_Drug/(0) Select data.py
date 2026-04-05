import pandas as pd

# 1. กำหนด Path ไฟล์
path_tested = r'C:\AMR_Thesis\MDR\DrugClass2\AllYears_DrugClass_tested.csv'
path_mapping = r'C:\AMR_Thesis\MDR\DrugClass2\Drug_class_for_MDR_new.csv'
output_path = r'C:\AMR_Thesis\MDR\model_for_1_Drug\AllYears_DrugClass_Exploded.csv'

# 2. โหลดข้อมูล
print("กำลังโหลดไฟล์...")
df_main = pd.read_csv(path_tested, low_memory=False)
df_map = pd.read_csv(path_mapping)

# --- ส่วนตรวจสอบชื่อคอลัมน์ (สำคัญมาก) ---
print("\n--- ตรวจสอบคอลัมน์ในไฟล์ AllYears_DrugClass_tested ---")
print(df_main.columns.tolist()) 
print("--------------------------------------------------\n")

# ให้คุณเปลี่ยนชื่อในเครื่องหมาย ' ' ด้านล่างนี้ให้ตรงกับที่ Print ออกมาเป๊ะๆ
# สมมติว่าเป็น 'ORGANISM_FULL' (ตัวพิมพ์ใหญ่ทั้งหมดตามที่คุณพิมพ์มาในแชทล่าสุด)
col_name_in_file = 'ORGANISM_FULL' 

# 3. Filter เฉพาะ Missing_Count == 0
if 'Missing_Count' in df_main.columns:
    df_main = df_main[df_main['Missing_Count'] == 0].copy()
else:
    # ถ้าหาชื่อ Missing_Count ไม่เจอ ลองเช็คตัวพิมพ์เล็กพิมพ์ใหญ่ด้วยครับ
    possible_missing = [c for c in df_main.columns if c.lower() == 'missing_count']
    if possible_missing:
        df_main = df_main[df_main[possible_missing[0]] == 0].copy()

# 4. เตรียม Mapping
mapping_dict = df_map.groupby('ORGANISM_WHO')['Antibiotic'].apply(list).to_dict()

def find_resistance_list(row):
    # ใช้ชื่อคอลัมน์ที่ระบุไว้ในข้อ 2
    try:
        org_name = str(row[col_name_in_file]).strip()
    except KeyError:
        return []

    if org_name not in mapping_dict:
        return []
    
    target_drugs = mapping_dict[org_name]
    res_list = []
    
    for drug in target_drugs:
        if drug in row.index:
            if str(row[drug]).strip().upper() == 'R':
                res_list.append(drug)
    return res_list

# 5. วิเคราะห์และแตกแถว
print("กำลังประมวลผล (Exploding)...")
df_main['resistant_drug_name'] = df_main.apply(find_resistance_list, axis=1)
df_exploded = df_main.explode('resistant_drug_name')

# 6. บันทึกผล
df_exploded.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"สำเร็จ! บันทึกไฟล์เรียบร้อยที่: {output_path}")
print(f"จำนวนแถวหลังแตกไฟล์: {len(df_exploded):,} แถว")