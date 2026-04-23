import pandas as pd

# กำหนด Path ของไฟล์ (ไฟล์ที่ผ่านการสร้าง sex_new และ ward_type_new มาแล้ว)
file_path = 'Correlation/MDR_for_correlation.csv'

try:
    # โหลดข้อมูล
    df = pd.read_csv(file_path, encoding='utf-8')
    print("✅ โหลดไฟล์ข้อมูลสำเร็จ กำลังเริ่มทำ One-Hot Encoding...")

    # ตรวจสอบว่ามีคอลัมน์ครบถ้วนหรือไม่
    cols_to_encode = ['age_group', 'sex_new', 'ward_type_new']
    missing_cols = [col for col in cols_to_encode if col not in df.columns]
    
    if not missing_cols:
        # 1. สร้างเฉพาะ Dummy Variables จากคอลัมน์ที่กำหนด
        df_dummies = pd.get_dummies(df[cols_to_encode], dtype=int)
        
        # 2. นำ Dummy Variables มาต่อท้าย DataFrame เดิม (ใช้ axis=1 เพื่อต่อในแนวคอลัมน์)
        df_encoded = pd.concat([df, df_dummies], axis=1)
        
        # บันทึกทับไฟล์เดิมเพื่อใช้เข้าโมเดลต่อไป
        df_encoded.to_csv(file_path, index=False, encoding='utf-8')
        print(f"✅ ทำ Dummy Variables และบันทึกทับไฟล์เรียบร้อยแล้ว (เก็บคอลัมน์เดิมไว้ครบถ้วน): {file_path}\n")
        
        # สุ่มดูคอลัมน์เดิมเทียบกับคอลัมน์ใหม่ที่ถูกสร้างขึ้น
        print("[Preview] ตัวอย่างคอลัมน์เดิม และ คอลัมน์ใหม่ที่พร้อมเข้าโมเดล (0/1):")
        
        # รวมชื่อคอลัมน์ต้นฉบับและคอลัมน์ดัมมี่เพื่อนำมาแสดงผล
        cols_to_show = cols_to_encode + list(df_dummies.columns)
        print(df_encoded[cols_to_show].head())
        
    else:
        print(f"⚠️ ไม่พบคอลัมน์ต่อไปนี้ในชุดข้อมูล: {missing_cols}")

except FileNotFoundError:
    print(f"❌ หาไฟล์ไม่พบ กรุณาตรวจสอบ Path: {file_path} อีกครั้ง")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")