import pandas as pd
import numpy as np

# กำหนด Path ของไฟล์ (บันทึกทับไฟล์เดิม)
file_path = 'Correlation/MDR_for_correlation.csv'

try:
    # โหลดข้อมูล
    df = pd.read_csv(file_path, encoding='utf-8')
    print("✅ โหลดไฟล์ข้อมูลสำเร็จ กำลังเริ่มกระบวนการจัดกลุ่ม...")

    # --- 1. จัดการคอลัมน์ ward_type ---
    if 'ward_type' in df.columns:
        # จัดการค่าว่าง (NaN) และลบช่องว่างหัวท้าย (spaces) ป้องกันข้อผิดพลาด
        ward_col = df['ward_type'].fillna('Unknow').astype(str).str.strip()
        
        # กำหนดเงื่อนไขการกรอง
        cond_ward_icu = ward_col.isin(['ICU', 'Icu', 'ccu', 'icu'])
        cond_ward_in = ward_col == 'in'
        cond_ward_out = ward_col == 'out'
        cond_ward_unk = ward_col.isin(['-', 'Unknow', 'nan', ''])
        
        # สร้างคอลัมน์ ward_type_new (หากไม่ตรงเงื่อนไขใดเลย จะตกไปที่ default='OTH')
        df['ward_type_new'] = np.select(
            [cond_ward_icu, cond_ward_in, cond_ward_out, cond_ward_unk],
            ['ICU', 'IN', 'OUT', 'Unknow'],
            default='OTH'
        )
    else:
        print("⚠️ ไม่พบคอลัมน์ 'ward_type' ในชุดข้อมูล")

    # --- 2. จัดการคอลัมน์ sex ---
    if 'sex' in df.columns:
        # จัดการค่าว่าง (NaN) และแปลงเป็น string
        sex_col = df['sex'].fillna('Unknow').astype(str).str.strip()
        
        # กำหนดเงื่อนไขการกรอง
        cond_sex_f = sex_col == 'f'
        cond_sex_m = sex_col.isin(['m', 'ช'])
        
        # สร้างคอลัมน์ sex_new (ค่าอื่น ๆ และค่าว่างที่เหลือ จะตกไปที่ default='Unknow' ทั้งหมด)
        df['sex_new'] = np.select(
            [cond_sex_f, cond_sex_m],
            ['F', 'M'],
            default='Unknow'
        )
    else:
        print("⚠️ ไม่พบคอลัมน์ 'sex' ในชุดข้อมูล")

    # --- 3. บันทึกทับไฟล์เดิม ---
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"✅ สร้างคอลัมน์และบันทึกทับไฟล์เดิมเรียบร้อยแล้ว: {file_path}\n")

    # --- 4. ตรวจสอบผลลัพธ์ (สุ่มดูเฉพาะค่าที่ซ้ำกันเพื่อเช็คความถูกต้อง) ---
    if 'ward_type' in df.columns:
        print("[Preview] การแปลงข้อมูล ward_type -> ward_type_new:")
        print(df[['ward_type', 'ward_type_new']].drop_duplicates().head(8))
        print("-" * 40)
        
    if 'sex' in df.columns:
        print("[Preview] การแปลงข้อมูล sex -> sex_new:")
        print(df[['sex', 'sex_new']].drop_duplicates().head(8))

except FileNotFoundError:
    print(f"❌ หาไฟล์ไม่พบ กรุณาตรวจสอบ Path: {file_path} อีกครั้ง")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")