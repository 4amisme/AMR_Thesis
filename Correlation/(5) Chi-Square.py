import pandas as pd
from scipy.stats import chi2_contingency

# กำหนด Path ของไฟล์
file_path = 'Correlation/MDR_for_correlation.csv'

try:
    # โหลดข้อมูล
    df = pd.read_csv(file_path, encoding='utf-8')
    print("✅ โหลดไฟล์ข้อมูลสำเร็จ กำลังเริ่มวิเคราะห์ Chi-Square...\n")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. การทดสอบปัจจัย Ward Type (เฉพาะ ICU, IN, OUT) กับ MDR
    # ---------------------------------------------------------
    if 'ward_type_new' in df.columns and 'MDR_status' in df.columns:
        df_ward = df[df['ward_type_new'].isin(['ICU', 'IN', 'OUT'])]
        ct_ward = pd.crosstab(df_ward['ward_type_new'], df_ward['MDR_status'])
        chi2_w, p_w, dof_w, expected_w = chi2_contingency(ct_ward)
        
        print("📌 1. ความสัมพันธ์ระหว่างประเภทหอผู้ป่วย (Ward) กับ MDR")
        print("ตารางความถี่ (Crosstab):")
        print(ct_ward.rename(columns={0: 'Non-MDR (0)', 1: 'MDR (1)'}))
        print(f"\nสถิติ Chi-Square: {chi2_w:.4f}")
        print(f"P-value: {p_w:.4e}")
        
        if p_w < 0.05:
            print("👉 สรุป: ประเภทหอผู้ป่วย มีความสัมพันธ์กับการเกิด MDR อย่างมีนัยสำคัญทางสถิติ")
        else:
            print("👉 สรุป: ประเภทหอผู้ป่วย ไม่มีความสัมพันธ์กับการเกิด MDR อย่างมีนัยสำคัญทางสถิติ")
        print("-" * 60)
    else:
        print("⚠️ ไม่พบคอลัมน์ 'ward_type_new' หรือ 'MDR_status'")

    # ---------------------------------------------------------
    # 2. การทดสอบปัจจัยเพศ (เฉพาะ F, M) กับ MDR
    # ---------------------------------------------------------
    if 'sex_new' in df.columns and 'MDR_status' in df.columns:
        df_sex = df[df['sex_new'].isin(['F', 'M'])]
        ct_sex = pd.crosstab(df_sex['sex_new'], df_sex['MDR_status'])
        chi2_s, p_s, dof_s, expected_s = chi2_contingency(ct_sex)
        
        print("\n📌 2. ความสัมพันธ์ระหว่างเพศ (Sex) กับ MDR")
        print("ตารางความถี่ (Crosstab):")
        print(ct_sex.rename(columns={0: 'Non-MDR (0)', 1: 'MDR (1)'}))
        print(f"\nสถิติ Chi-Square: {chi2_s:.4f}")
        print(f"P-value: {p_s:.4e}")
        
        if p_s < 0.05:
            print("👉 สรุป: เพศ มีความสัมพันธ์กับการเกิด MDR อย่างมีนัยสำคัญทางสถิติ")
        else:
            print("👉 สรุป: เพศ ไม่มีความสัมพันธ์กับการเกิด MDR อย่างมีนัยสำคัญทางสถิติ")
        print("-" * 60)
    else:
        print("⚠️ ไม่พบคอลัมน์ 'sex_new' หรือ 'MDR_status'")

    # ---------------------------------------------------------
    # 3. การทดสอบปัจจัยช่วงอายุ (Age Group) กับ MDR
    # ---------------------------------------------------------
    if 'age_group' in df.columns and 'MDR_status' in df.columns:
        target_ages = ['1-4', '15-24', '25-34', '35-44', '45-54', 
                       '5-14', '55-64', '65-74', '75-84', '85+', '<1', 'Unknown']
        
        df_age = df[df['age_group'].isin(target_ages)]
        ct_age = pd.crosstab(df_age['age_group'], df_age['MDR_status'])
        
        # จัดเรียง Index ของตารางให้ดูง่ายขึ้น (เรียงตามอายุน้อยไปมาก)
        ordered_age = ['<1', '1-4', '5-14', '15-24', '25-34', '35-44', 
                       '45-54', '55-64', '65-74', '75-84', '85+', 'Unknown']
        # เลือกเฉพาะช่วงอายุที่มีข้อมูลจริงในตาราง
        ordered_age = [age for age in ordered_age if age in ct_age.index]
        ct_age = ct_age.reindex(ordered_age)
        
        chi2_a, p_a, dof_a, expected_a = chi2_contingency(ct_age)
        
        print("\n📌 3. ความสัมพันธ์ระหว่างช่วงอายุ (Age Group) กับ MDR")
        print("ตารางความถี่ (Crosstab):")
        print(ct_age.rename(columns={0: 'Non-MDR (0)', 1: 'MDR (1)'}))
        print(f"\nสถิติ Chi-Square: {chi2_a:.4f}")
        print(f"P-value: {p_a:.4e}")
        
        if p_a < 0.05:
            print("👉 สรุป: ช่วงอายุ มีความสัมพันธ์กับการเกิด MDR อย่างมีนัยสำคัญทางสถิติ")
        else:
            print("👉 สรุป: ช่วงอายุ ไม่มีความสัมพันธ์กับการเกิด MDR อย่างมีนัยสำคัญทางสถิติ")
        print("=" * 60)
    else:
        print("⚠️ ไม่พบคอลัมน์ 'age_group' หรือ 'MDR_status'")

except FileNotFoundError:
    print(f"❌ หาไฟล์ไม่พบ กรุณาตรวจสอบ Path: {file_path}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")