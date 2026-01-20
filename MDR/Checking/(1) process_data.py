import pandas as pd
import os

def process_data_final():
    # ==========================================
    # 1. ตั้งค่าไฟล์ Input / Output
    # ==========================================
    # Path ข้อมูล: MDR -> data -> AllYears_processed.csv
    input_file = os.path.join("MDR", "data", "AllYears_processed.csv")
    output_file = "Checking.csv"

    print(f"[Step 2] กำลังอ่านไฟล์ข้อมูลจาก: {input_file}")

    # ตรวจสอบว่ามีไฟล์ Input หรือไม่
    if not os.path.exists(input_file):
        print(f"❌ Error: ไม่พบไฟล์ {input_file}")
        print("คำแนะนำ: ตรวจสอบว่าไฟล์ 'AllYears_processed.csv' อยู่ในโฟลเดอร์ 'MDR/data' ถูกต้องหรือไม่")
        return

    # อ่านไฟล์ (ใช้ low_memory=False เพื่อป้องกัน Warning กรณีไฟล์ใหญ่)
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"อ่านไฟล์ไม่ได้: {e}")
        return

    # ==========================================
    # 🌟 จุดสำคัญ: แก้ปัญหาชื่อยาหาไม่เจอ (The Fix)
    # ==========================================
    # 1. .strip() -> ตัดช่องว่างที่ซ่อนอยู่หน้า-หลัง (แก้ปัญหา 'ceftriaxone ')
    # 2. .lower() -> แปลงเป็นตัวพิมพ์เล็กทั้งหมด (แก้ปัญหา 'Ceftriaxone')
    df.columns = df.columns.str.strip().str.lower()
    print("✅ ทำความสะอาดชื่อคอลัมน์ (ตัดช่องว่าง + แปลงตัวเล็ก) เรียบร้อยแล้ว")

    # ==========================================
    # 2. รายชื่อยา (Target Antibiotics)
    # ==========================================
    target_antibiotics = [
        'amikacin',
        'amoxicillin/clavulanic acid',
        'ampicillin',
        'ampicillin/sulbactam',
        'cefazolin',
        'cefepime',
        'cefoperazone/sulbactam',
        'cefotaxime',
        'cefoxitin',
        'ceftazidime',
        'ceftriaxone',
        'cefuroxime',
        'chloramphenicol',
        'ciprofloxacin',
        'clindamycin',
        'ertapenem',
        'erythromycin',
        'fosfomycin',
        'gentamicin',
        'gentamicin high',
        'imipenem',
        'levofloxacin',
        'meropenem',
        'nitrofurantoin',
        'norfloxacin',
        'oxacillin',
        'penicillin',
        'piperacillin/tazobactam',
        'teicoplanin',
        'tetracycline',
        'trimethoprim/sulfamethoxazole',
        'vancomycin',
        'colistin'
    ]

    # ตรวจสอบว่ายาที่เราต้องการ มีอยู่ในไฟล์จริงกี่ตัว
    available_antibiotics = [col for col in target_antibiotics if col in df.columns]
    
    # แจ้งเตือนยาที่หาไม่เจอ (ถ้าลิสต์นี้ว่างเปล่า แสดงว่าเจอครบ)
    missing = set(target_antibiotics) - set(available_antibiotics)
    if missing:
        print(f"⚠️ คำเตือน: ยังคงไม่พบคอลัมน์ยาเหล่านี้ในไฟล์: {missing}")
    else:
        print("✅ เยี่ยมมาก! พบครบทุกคอลัมน์ยาที่ต้องการ")

    print(f"กำลังประมวลผลข้อมูลยาจำนวน {len(available_antibiotics)} ชนิด...")

    # ==========================================
    # 3. แปลงข้อมูล (Melt)
    # ==========================================
    try:
        # ใช้ 'organism_full' ตัวเล็ก (เพราะเราสั่ง lower() ไปแล้ว)
        id_col = 'organism_full' 
        
        if id_col not in df.columns:
            print(f"❌ Error: ไม่พบคอลัมน์ '{id_col}' ในไฟล์")
            return

        # Unpivot ข้อมูล (เปลี่ยนแนวกว้างเป็นแนวยาว)
        melted_df = df.melt(
            id_vars=[id_col], 
            value_vars=available_antibiotics, 
            var_name='antibiotic', 
            value_name='result'
        )
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการแปลงข้อมูล: {e}")
        return

    # ==========================================
    # 4. กรองและทำความสะอาดข้อมูล (Filter)
    # ==========================================
    print("กำลังคัดกรองเฉพาะผลลัพธ์ R, I, S...")
    
    # แปลงผลลัพธ์เป็นตัวพิมพ์ใหญ่ (ป้องกันกรณีเจอ 's' เล็ก) และเป็น string
    melted_df['result'] = melted_df['result'].astype(str).str.upper().str.strip()
    
    # เลือกเอาเฉพาะ R, I, S
    filtered_df = melted_df[melted_df['result'].isin(['R', 'I', 'S'])]

    # เปลี่ยนชื่อคอลัมน์ให้ตรงตามโจทย์ (organism_full -> organism)
    filtered_df = filtered_df.rename(columns={id_col: 'organism'})

    # ==========================================
    # 5. สรุปยอด (Group By & Count)
    # ==========================================
    print("กำลังคำนวณจำนวน (Group By & Count)...")
    
    # SQL Logic: SELECT ..., COUNT(*) FROM ... GROUP BY organism, antibiotic, result
    summary_df = filtered_df.groupby(['organism', 'antibiotic', 'result']).size().reset_index(name='total_count')
    
    # SQL Logic: ORDER BY organism ASC, antibiotic ASC
    summary_df = summary_df.sort_values(by=['organism', 'antibiotic'])

    # ==========================================
    # 6. บันทึกไฟล์
    # ==========================================
    print(f"กำลังบันทึกไฟล์ไปที่: {output_file}")
    summary_df.to_csv(output_file, index=False)
    print("🎉 เสร็จสมบูรณ์! เช็คไฟล์ Checking.csv ได้เลยครับ")

if __name__ == "__main__":
    process_data_final()