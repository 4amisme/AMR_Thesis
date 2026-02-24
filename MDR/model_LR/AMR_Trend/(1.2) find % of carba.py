import pandas as pd
import os
import numpy as np

# เดี๋ยวมาทำต่อ จะต้องทำเป็นกราฟเส้นของการดื้อยาแต่ละคลาสมาดูว่าเป็นยังไงก่อน + reorder columns

# กำหนด Path ให้ตรงกับโฟลเดอร์ที่คุณใช้งานอยู่
INPUT_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend/amr-a_baumannii_selected.csv'
BASE_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend'

def step2_prepare_monthly():
    print("⏳ Step 2: Extracting Month from spec_date and Preparing Monthly %R...")
    
    # 1. สร้างโครงสร้างโฟลเดอร์ย่อยรอไว้
    folders = ['All data', 'by ward', 'by specimen']
    for folder in folders:
        for sub in ['Data', 'Seasonality', 'Forecasts']:
            os.makedirs(os.path.join(BASE_DIR, folder, sub), exist_ok=True)
            
    # 2. โหลดข้อมูลที่คลีนแล้วจาก Step 1
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    
    # ==========================================
    # 🌟 3. จัดการคอลัมน์เวลา (ดึงเดือนจาก spec_date)
    # ==========================================
    if 'spec_date' not in df.columns:
        print("❌ ไม่พบคอลัมน์ 'spec_date' ในไฟล์ข้อมูล กรุณาตรวจสอบไฟล์ Step 1")
        return
        
    # แปลง spec_date เป็นรูปแบบ Datetime (ตัวไหนพังจะกลายเป็น NaT)
    df['spec_date'] = pd.to_datetime(df['spec_date'], errors='coerce')
    
    # ตัดแถวที่ไม่มีวันที่ส่งตรวจทิ้งไป
    df = df.dropna(subset=['spec_date'])
    
    # กรองให้เหลือแค่ปี 2015-2024 (เพื่อความปลอดภัย)
    df = df[(df['spec_date'].dt.year >= 2015) & (df['spec_date'].dt.year <= 2024)].copy()
    
    # สกัด x_month และ x_year ออกมา
    df['x_year'] = df['spec_date'].dt.year
    df['x_month'] = df['spec_date'].dt.month
    
    # สร้างคอลัมน์ date (YYYY-MM-01) เพื่อใช้เป็นแกนเวลาของ Time Series
    df['date'] = pd.to_datetime(df['x_year'].astype(str) + '-' + df['x_month'].astype(str).str.zfill(2) + '-01')

    # ==========================================
    # 4. ฟังก์ชันสำหรับคำนวณ %R ของ Imipenem และ Meropenem
    # ==========================================
    def calculate_percent_r(data, group_cols):
        results = []
        for drug in ['imipenem', 'meropenem']:
            if drug not in data.columns:
                continue
            
            # ตัดบรรทัดที่ไม่ได้ผลตรวจยานี้ทิ้ง
            df_drug = data.dropna(subset=[drug]).copy()
            
            # กำหนด: ถ้าผลเป็น 'R' คือดื้อยา (1), ถ้าเป็น S หรือ I คือไม่ดื้อ (0)
            df_drug['is_resistant'] = df_drug[drug].astype(str).str.upper().apply(lambda x: 1 if x == 'R' else 0)
            
            # นับยอดรวมของเดือนนั้นๆ
            agg_df = df_drug.groupby(group_cols).agg(
                n_tested=('is_resistant', 'count'),
                n_resistant=('is_resistant', 'sum')
            ).reset_index()
            
            # คำนวณเปอร์เซ็นต์
            agg_df['drug'] = drug.capitalize()
            agg_df['percent_R'] = ((agg_df['n_resistant'] / agg_df['n_tested']) * 100).round(2)
            results.append(agg_df)
            
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    # ==========================================
    # 5. เริ่มคำนวณแยกตาม 3 หมวดหมู่ แล้วบันทึกไฟล์
    # ==========================================
    
    # [5.1] หมวด All data (ภาพรวมประเทศ)
    overall_df = calculate_percent_r(df, ['date'])
    overall_df.to_csv(os.path.join(BASE_DIR, 'All data', 'Data', 'monthly_overall.csv'), index=False)
    print("✅ บันทึกไฟล์สำเร็จ: All data/Data/monthly_overall.csv")
    
    # [5.2] หมวด By ward (เฉพาะ icu, in, out)
    df_ward = df[df['ward_type'].isin(['icu', 'in', 'out'])].copy()
    ward_df = calculate_percent_r(df_ward, ['date', 'ward_type'])
    ward_df.to_csv(os.path.join(BASE_DIR, 'by ward', 'Data', 'monthly_ward.csv'), index=False)
    print("✅ บันทึกไฟล์สำเร็จ: by ward/Data/monthly_ward.csv")
    
    # [5.3] หมวด By specimen (Top 3 + other)
    # สมมติว่าใน Step 1 เราเตรียมคอลัมน์ spec_group ไว้แล้ว
    df_spec = df.dropna(subset=['spec_group']).copy()
    spec_df = calculate_percent_r(df_spec, ['date', 'spec_group'])
    spec_df.to_csv(os.path.join(BASE_DIR, 'by specimen', 'Data', 'monthly_specimen.csv'), index=False)
    print("✅ บันทึกไฟล์สำเร็จ: by specimen/Data/monthly_specimen.csv")
    
    print(f"\n🎉 Step 2 Complete: ข้อมูลรายเดือนพร้อมลุย Time Series แล้วครับ!")

if __name__ == "__main__":
    step2_prepare_monthly()