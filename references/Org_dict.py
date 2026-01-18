import pandas as pd
# กำหนดชื่อไฟล์ CSV
input_file = 'references/org_to_map.csv'       # ชื่อไฟล์ต้นทาง
output_file = 'mapped_org.csv' # ชื่อไฟล์ปลายทางที่จะบันทึก

# อ่านไฟล์ CSV
df = pd.read_csv(input_file, encoding='utf-8')

# แปลงข้อความที่มี ',' คั่น ให้เป็น List
df['Code_Org'] = df['Code_Org'].astype(str).str.split(',')

# Explode ข้อมูลลงมาเป็นบรรทัดใหม่
df_exploded = df.explode('Code_Org')

# ลบช่องว่างหน้า-หลังที่อาจติดมา
df_exploded['Code_Org'] = df_exploded['Code_Org'].str.strip()

df_exploded.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"เสร็จเรียบร้อย ข้อมูลถูกบันทึกที่ไฟล์: {output_file}")
print("ตัวอย่างข้อมูล:")
print(df_exploded.head())