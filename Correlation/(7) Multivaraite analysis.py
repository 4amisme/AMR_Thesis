import pandas as pd
import statsmodels.api as sm
import numpy as np

# 1. โหลดข้อมูล
file_path = 'Correlation/MDR_for_correlation.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 2. กำหนดตัวแปรต้น (X) และตัวแปรตาม (y)
# เลือกคอลัมน์ที่เป็น Dummy (0/1) โดยต้อง "ตัดกลุ่มอ้างอิงออก 1 กลุ่ม" ในแต่ละปัจจัย
# เราจะตัด sex_new_F, ward_type_new_OUT และ age_group_15-24 ออก

predictors = [
    # ปัจจัยเพศ (เทียบกับ F)
    'sex_new_M', 
    
    # ปัจจัยวอร์ด (เทียบกับ OUT)
    'ward_type_new_ICU', 'ward_type_new_IN', 'ward_type_new_OTH',
    
    # ปัจจัยอายุ (เทียบกับ 15-24)
    'age_group_<1', 'age_group_1-4', 'age_group_5-14', 'age_group_25-34', 
    'age_group_35-44', 'age_group_45-54', 'age_group_55-64', 'age_group_65-74', 
    'age_group_75-84', 'age_group_85+'
]

y = df['MDR_status']
X = df[predictors]

# 3. เพิ่มค่า Constant (Intercept) ให้กับโมเดล
X = sm.add_constant(X)

# 4. สร้างและ Fit โมเดล Logistic Regression
model = sm.Logit(y, X).fit()

# 5. สรุปผลลัพธ์และคำนวณ Adjusted Odds Ratio (aOR)
summary = model.summary2().tables[1] # ดึงตารางผลลัพธ์ออกมาจัดการต่อ
summary['OR (aOR)'] = np.exp(summary['Coef.']) # แปลง Log Odds เป็น Odds Ratio
summary['95% CI Lower'] = np.exp(summary['[0.025'])
summary['95% CI Upper'] = np.exp(summary['0.975]'])

# กรองเฉพาะคอลัมน์ที่จำเป็นสำหรับการรายงานผลในธีสิส
result_table = summary[['OR (aOR)', '95% CI Lower', '95% CI Upper', 'P>|z|']]

print("\n--- Multivariate Logistic Regression Results ---")
print(result_table.round(3))

# 6. บันทึกผลลัพธ์ลงไฟล์ Excel หรือ CSV เพื่อนำไปเขียนรายงาน
result_table.to_csv('Correlation/Logistic_Regression_Results.csv')