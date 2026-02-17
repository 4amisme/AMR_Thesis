import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import pandas as pd


# เลือกกลุ่มยาที่น่าสนใจมาเปรียบเทียบ (อ้างอิงจากผลลัพธ์ของคุณ)
# 1. กลุ่มที่มีค่า Seasonal Strength สูงสุด (อันดับ 4)
# 2. กลุ่มที่มีค่า Seasonal Strength ต่ำ (อันดับ 5 หรือ 2)

target_classes = [
    'AMINOGLYCOSIDES, CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS', # เปลี่ยนเป็นชื่อเต็มในไฟล์ของคุณ
    'AMINOGLYCOSIDES, CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, β-LACTAM COMBINATION AGENTS',
    'CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS',
    'CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, β-LACTAM COMBINATION AGENTS',
    'CARBAPENEMS, CEPHEMS, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS'
]

def plot_seasonality_analysis(df, drug_class_name):
    # ตรวจสอบว่ามีคอลัมน์นี้หรือไม่ (เนื่องจากชื่ออาจถูกตัด)
    actual_col = [c for c in df.columns if c.startswith(drug_class_name[:20])]
    if not actual_col:
        return
    
    series = df[actual_col[0]]
    # สร้าง Index วันที่สำหรับการ Plot
    dates = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    
    # ทำ Decomposition
    res = STL(series, period=12, robust=True).fit()
    
    # สร้างกราฟ
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # กราฟข้อมูลจริง (Observed)
    ax1.plot(dates, series, color='blue', label='Observed (Actual %)')
    ax1.set_title(f'Analysis for: {drug_class_name}', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # กราฟแนวโน้ม (Trend)
    ax2.plot(dates, res.trend, color='red', label='Trend (Overall Direction)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # กราฟฤดูกาล (Seasonal) - จุดนี้จะบอกว่ามันซ้ำเดิมแค่ไหน
    ax3.plot(dates, res.seasonal, color='green', label='Seasonal (Yearly Pattern)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# เรียกใช้ฟังก์ชัน (สมมติว่า final_df คือตาราง Wide Format ที่เติมครบ 120 เดือนแล้ว)
# for cls in drug_classes:
#     plot_seasonality_analysis(final_df, cls)