import pandas as pd
import os

def process_data_safe():
    input_file = "C:\AMR_Thesis\MDR\data\AllYears_processed.csv"   # หรือชื่อไฟล์ CSV ของคุณ
    output_file = "Checking.csv"

    print(f"กำลังอ่านไฟล์ {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: ไม่พบไฟล์ {input_file}")
        return

    # 1. อ่านข้อมูล (ระบุ dtype เพื่อลดการกินเมมโมรี่)
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"อ่านไฟล์ไม่ได้: {e}")
        return

    # 2. ระบุชื่อยาที่ต้องการจริงๆ (ตามที่คุณเคยให้มา)
    # วิธีนี้ปลอดภัยที่สุด ไม่กินแรมฟรีๆ กับคอลัมน์ขยะ
    target_antibiotics = [
        'amikacin', 'amoxicillin/clavulanic acid', 'ampicillin', 'ampicillin/sulbactam',
        'cefazolin', 'cefepime', 'cefoperazone/sulbactam', 'cefotaxime', 'cefoxitin',
        'ceftazidime', 'ceftriaxone', 'cefuroxime', 'chloramphenicol', 'ciprofloxacin',
        'clindamycin', 'ertapenem', 'erythromycin', 'fosfomycin', 'gentamicin',
        'gentamicin high', 'imipenem', 'levofloxacin', 'meropenem', 'nitrofurantoin',
        'norfloxacin', 'oxacillin', 'penicillin', 'piperacillin/tazobactam',
        'teicoplanin', 'tetracycline', 'trimethoprim/sulfamethoxazole', 'vancomycin', 'colistin'
    ]

    # ตรวจสอบว่าในไฟล์มีคอลัมน์ยาตัวไหนบ้าง (กัน Error กรณีไฟล์จริงชื่อไม่ตรงเป๊ะ)
    # เราจะเอาเฉพาะตัวที่มีอยู่ในไฟล์จริงๆ เท่านั้น
    available_antibiotics = [col for col in target_antibiotics if col in df.columns]
    
    missing_cols = set(target_antibiotics) - set(available_antibiotics)
    if missing_cols:
        print(f"แจ้งเตือน: ไม่พบคอลัมน์ยาเหล่านี้ในไฟล์ (จะถูกข้ามไป): {missing_cols}")

    print(f"กำลังประมวลผลยาจำนวน {len(available_antibiotics)} ชนิด...")

    # 3. แปลงข้อมูล (Melt) เฉพาะคอลัมน์ที่ระบุเท่านั้น
    try:
        melted_df = df.melt(
            id_vars=['ORGANISM_FULL'],       # คอลัมน์หลัก
            value_vars=available_antibiotics, # เฉพาะรายชื่อยาที่ระบุ
            var_name='antibiotic', 
            value_name='result'
        )
    except KeyError as e:
        print(f"Error: ชื่อคอลัมน์ ORGANISM_FULL อาจไม่ถูกต้อง หรือไม่มีในไฟล์: {e}")
        return

    # 4. กรองเอาเฉพาะผล R, I, S
    print("กำลังคัดกรองผลลัพธ์...")
    final_df = melted_df[melted_df['result'].isin(['R', 'I', 'S'])]

    # เปลี่ยนชื่อคอลัมน์ให้ตรงตามโจทย์
    final_df = final_df.rename(columns={'ORGANISM_FULL': 'organism'})

    # 5. บันทึกไฟล์
    print(f"กำลังบันทึกไฟล์ {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("เสร็จสมบูรณ์! (Memory Safe Version)")

if __name__ == "__main__":
    process_data_safe()