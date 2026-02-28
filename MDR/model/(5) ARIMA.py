import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ปิดแจ้งเตือน Warning เพื่อความสะอาดของ Output
warnings.simplefilter("ignore")

# ==========================================
# 1. ฟังก์ชันคำนวณ Metrics
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE สำหรับวัดประสิทธิภาพ"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    return round(rmse, 4), round(wape, 4)

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์และทำนาย (ARIMA)
# ==========================================

def run_mdr_forecasting_arima(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}")
    print(f"Analyzing (ARIMA): {target_drug_name}")
    print(f"{'='*50}")

    # --- [A] การแบ่งข้อมูล (80/20) ปรับให้เหมือน SARIMA ---
    n = len(series)
    train_size = int(n * 0.80) 
    train_data = series.iloc[:train_size]
    test_data = series.iloc[train_size:]

    # --- [B] วิเคราะห์ ACF / PACF ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(series, lags=24, ax=ax1)
    ax1.set_title(f'ACF: {target_drug_name}')
    plot_pacf(series, lags=24, ax=ax2)
    ax2.set_title(f'PACF: {target_drug_name}')
    plt.show()

    # --- [C] Auto-ARIMA Parameter Tuning (ใช้แค่ Train Data) ---
    print("\n[กำลังค้นหาค่า Parameter ที่ดีที่สุด (p, d, q)...]")
    
    # สำหรับ ARIMA ปกติ เราจะปิด seasonal=False และไม่ใช้ P, D, Q, m
    stepwise_model = auto_arima(
        train_data,                 # ⚠️ ใช้แค่ train_data ป้องกัน Data Leakage
        start_p=0, start_q=0,
        max_p=5, max_q=5, 
        d=None,                     # ให้โมเดลหาค่า d อัตโนมัติ
        seasonal=False,             # ⚠️ จุดสำคัญ: ปิดโหมดฤดูกาลสำหรับ ARIMA
        stepwise=True, 
        suppress_warnings=True, 
        error_action='ignore', 
        trace=True
    )

    best_order = stepwise_model.order
    print(f"\n>>> Best ARIMA Order: {best_order}")
    print(f">>> Best AIC: {stepwise_model.aic():.2f}")

    # --- [D] Model Training & Forecasting (กระบวนการ 2 ขั้นตอน) ---
    
    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
    # เทรนเพื่อวัดผลด้วย Train Data
    model_eval = ARIMA(train_data, order=best_order).fit()
    
    # ทำนายช่วง Test Data
    test_pred_arima = model_eval.forecast(steps=len(test_data))
    rmse, wape = calculate_metrics(test_data, test_pred_arima)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (100% Data) ---")
    # เทรนโมเดลตัวจริงด้วยข้อมูลทั้งหมด (100%) เพื่อทำนายอนาคต
    final_model = ARIMA(series, order=best_order).fit()
    
    # ทำนายล่วงหน้า 5 ปี
    forecast_arima = final_model.forecast(steps=forecast_months)

    # --- [E] การพล็อตแสดงผล ---
    
    plt.figure(figsize=(12, 6))
    
    # พล็อตข้อมูลจริง
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data (2015-2024)', linewidth=1.5)
    
    # สร้างเส้นพยากรณ์อนาคตให้เชื่อมต่อกันสนิท
    forecast_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    forecast_val = np.concatenate([[series.values[-1]], forecast_arima.values])
    
    plt.plot(forecast_idx, forecast_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'ARIMA {best_order} Forecast (Next 5 years)', linewidth=1.5)

    # ตกแต่งกราฟ
    plt.title(f'MDR Pattern Prediction: {target_drug_name}', fontsize=13, pad=15)
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

file_path = os.path.join("MDR", "model", "a_baumannii_ur.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left').fillna(0)
    final_df.index = all_months

    target_drug = 'CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        run_mdr_forecasting_arima(series_data, "Acinetobacter baumannii")
    else:
        print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")