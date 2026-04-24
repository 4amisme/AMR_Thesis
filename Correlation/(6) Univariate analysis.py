import pandas as pd
import statsmodels.api as sm
import numpy as np

# 1. กำหนด Path ของไฟล์
file_path = 'Correlation/MDR_for_correlation.csv'

try:
    # โหลดข้อมูล
    df = pd.read_csv(file_path, encoding='utf-8')
    print("✅ โหลดไฟล์ข้อมูลสำเร็จ กำลังรันโมเดล Multivariate Logistic Regression...")

    # 2. กำหนดตัวแปรตาม (y)
    y = df['MDR_status']

    # 3. กำหนดตัวแปรต้น (X) โดยนำตัวแปรทั้งหมดมาใส่พร้อมกัน
    # ข้อควรระวัง: ไม่ใส่กลุ่มอ้างอิง (sex_new_F, ward_type_new_OUT, age_group_15-24) ลงในลิสต์นี้
    predictors = [
        'sex_new_M', 
        'ward_type_new_ICU', 'ward_type_new_IN', 'ward_type_new_OTH',
        'age_group_<1', 'age_group_1-4', 'age_group_5-14', 'age_group_25-34', 
        'age_group_35-44', 'age_group_45-54', 'age_group_55-64', 'age_group_65-74', 
        'age_group_75-84', 'age_group_85+'
    ]
    
    # ดึงข้อมูลเฉพาะคอลัมน์ที่ต้องการ
    X = df[predictors]

    # เพิ่มค่า Constant (Intercept) เข้าไปในสมการ (จำเป็นสำหรับสมการ Logistic)
    X = sm.add_constant(X)

    # 4. สร้างและ Fit โมเดล
    # ใช้ disp=False เพื่อไม่ให้แสดง log การคำนวณที่รกหน้าจอ
    model = sm.Logit(y, X).fit(disp=False)

    # 5. ดึงผลลัพธ์และคำนวณ aOR
    params = model.params
    conf = model.conf_int()
    pvalues = model.pvalues

    # สร้าง List เพื่อเก็บผลลัพธ์
    results_list = []

    for var in params.index:
        # ข้ามการแสดงผลค่า const (Intercept) ในตารางสุดท้าย เพราะไม่ต้องใช้รายงานผล
        if var == 'const':
            continue
            
        aOR = np.exp(params[var])
        ci_lower = np.exp(conf.loc[var, 0])
        ci_upper = np.exp(conf.loc[var, 1])
        pval = pvalues[var]
        
        results_list.append({
            'Factor': var,
            'Adjusted OR (aOR)': round(aOR, 3),
            '95% CI Lower': round(ci_lower, 3),
            '95% CI Upper': round(ci_upper, 3),
            'P-value': pval
        })

    # 6. สร้าง DataFrame สำหรับแสดงผล
    df_multi_results = pd.DataFrame(results_list)

    # จัดรูปแบบ P-value ให้ดูสวยงามเป็นทางการ
    df_multi_results['P-value'] = df_multi_results['P-value'].apply(
        lambda x: '< 0.001' if x < 0.001 else round(x, 3)
    )

    print("\n--- ตารางผลลัพธ์ Multivariate Logistic Regression ---")
    print(df_multi_results.to_string(index=False))

    # 7. บันทึกผลลัพธ์ลงไฟล์ CSV
    output_path = 'Correlation/Multivariate_Logistic_Results.csv'
    df_multi_results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ บันทึกไฟล์ผลลัพธ์เรียบร้อยแล้วที่: {output_path}")

except FileNotFoundError:
    print(f"❌ หาไฟล์ไม่พบ กรุณาตรวจสอบ Path: {file_path}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")