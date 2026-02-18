import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf

# ==========================================
# 1. ฟังก์ชันคำนวณ Metrics
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE สำหรับวัดประสิทธิภาพ"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    return round(rmse, 4), round(wape, 4)

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์และทำนาย (Auto-ARIMA Version)
# ==========================================

def run_mdr_forecasting_auto_arima(series, target_drug_name, forecast_months=60):
    # --- [A] การแบ่งข้อมูล (90/10 เพื่อให้มีข้อมูล Train มากขึ้นสำหรับ Auto-ARIMA) ---
    n = len(series)
    val_end = int(n * 0.90) 
    train = series.iloc[:val_end]
    test = series.iloc[val_end:]

    # --- [B] วิเคราะห์ ACF (ใช้ข้อมูลทั้งหมด) ---
    plt.figure(figsize=(10, 4))
    plot_acf(series, lags=24, ax=plt.gca())
    plt.title(f'Autocorrelation (ACF): {target_drug_name}')
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- [C] Auto-ARIMA Model Training & Parameter Tuning ---
    print(f"\n[กำลังค้นหาค่า Parameter ที่ดีที่สุดสำหรับ {target_drug_name}...]")
    
    # ค้นหา p, d, q อัตโนมัติโดยใช้ค่า AIC เป็นเกณฑ์
    model_auto = auto_arima(series, 
                            start_p=0, start_q=0,
                            max_p=5, max_q=5, 
                            d=None,           # ให้โมเดลเลือกค่า d ที่เหมาะสมเอง
                            seasonal=False,   # ปิด Seasonal หากต้องการ ARIMA ปกติ
                            stepwise=True, 
                            suppress_warnings=True, 
                            error_action='ignore', 
                            trace=True)

    best_order = model_auto.order
    print(f"\nBest ARIMA Order: {best_order}")
    print(model_auto.summary())

    # --- [D] Forecasting & Evaluation ---
    
    # 1. ทำนายอนาคต (Forecast)
    forecast_values = model_auto.predict(n_periods=forecast_months)
    
    # 2. ทำนายย้อนหลังในช่วง Test set เพื่อวัดผล
    # หมายเหตุ: predict_in_sample จะให้ค่าที่แม่นยำในการวัดผล Performance
    test_pred = model_auto.predict_in_sample(start=val_end, end=n-1)
    rmse, wape = calculate_metrics(test, test_pred)
    
    print("-" * 50)
    print(f"Auto-ARIMA Evaluation Result (Order: {best_order}):")
    print(f"RMSE: {rmse}")
    print(f"WAPE: {wape}%")
    print("-" * 50)

    # --- [E] การพล็อตแสดงผล ---
    plt.figure(figsize=(12, 6))
    
    # พล็อตข้อมูลจริง (สีน้ำเงิน)
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data (2015-2024)', linewidth=1.5)
    
    # สร้าง Index สำหรับช่วงเวลาที่ทำนายอนาคต
    forecast_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), 
                                   periods=forecast_months, freq='MS')
    
    # เชื่อมจุดสุดท้ายของข้อมูลจริงกับจุดแรกของ Forecast เพื่อความต่อเนื่องของกราฟ
    connection_idx = [series.index[-1]] + list(forecast_index)
    connection_val = [series.values[-1]] + list(forecast_values)
    
    plt.plot(connection_idx, connection_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'Auto-ARIMA {best_order} Forecast (Next 5 years)', linewidth=1.5)

    # ตกแต่งกราฟ
    plt.title(f'Auto-ARIMA Forecasting: {target_drug_name}\nBest Parameters: {best_order}', fontsize=13, pad=15)
    plt.xlabel('Year')
    plt.ylabel('Resistance Percentage (%R)')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. ส่วนการรันข้อมูล
# ==========================================

# ปรับ Path ตามโครงสร้างโฟลเดอร์ของคุณ
file_path = os.path.join("MDR", "model", "acinetobacter_baumannii.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    # เตรียมข้อมูล Wide Format
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    
    # สร้าง Index วันที่ให้สมบูรณ์
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left').fillna(0)
    final_df.index = all_months

    # ระบุกลุ่มยาที่ต้องการวิเคราะห์
    target_drug = 'AMINOGLYCOSIDES, CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        # เรียกใช้งานฟังก์ชัน Auto-ARIMA
        run_mdr_forecasting_auto_arima(series_data, "Acinetobacter baumannii")
    else:
        print(f"ไม่พบกลุ่มยาในไฟล์: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่ path: {os.path.abspath(file_path)}")