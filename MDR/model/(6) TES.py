import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import itertools
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import warnings

# ปิดการแจ้งเตือนเพื่อความสะอาดของ Output
warnings.filterwarnings("ignore")

# ==========================================
# 1. ฟังก์ชันสำหรับการคำนวณ Metrics และ Tuning
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE สำหรับวัดประสิทธิภาพ"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    return round(rmse, 4), round(wape, 4)

def grid_search_tes(train_data):
    """ทำ Parameter Tuning เพื่อหา Configuration ที่ดีที่สุดด้วยเกณฑ์ AIC"""
    trend_opts = ['add', None]      # ใช้แบบ Additive เหมาะกับข้อมูลเปอร์เซ็นต์
    seasonal_opts = ['add', None]   # ใช้ Additive สำหรับฤดูกาล
    damped_opts = [True, False]     # หางค้าง (ลดความชันเทรนด์ในอนาคต)
    
    best_aic = float('inf')
    best_config = None
    
    # สร้างรายการการตั้งค่าทั้งหมดที่เป็นไปได้
    configs = list(itertools.product(trend_opts, seasonal_opts, damped_opts))
    
    for t, s, d in configs:
        # ข้ามลอจิกที่เป็นไปไม่ได้: ถ้าไม่มี Trend จะทำ Damped Trend ไม่ได้
        if t is None and d is True:
            continue
            
        try:
            # เทรนและหาค่า AIC บนชุดข้อมูล Train
            model = ExponentialSmoothing(
                train_data, trend=t, seasonal=s, seasonal_periods=12, damped_trend=d
            ).fit(optimized=True)
            
            if model.aic < best_aic:
                best_aic = model.aic
                best_config = (t, s, d)
        except:
            continue
            
    return best_config

# ==========================================
# 2. ฟังก์ชันหลักสำหรับการวิเคราะห์และทำนาย
# ==========================================

def run_mdr_forecasting_tes(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}")
    print(f"Analyzing (Triple ES / Holt-Winters): {target_drug_name}")
    print(f"{'='*50}")

    # --- [A] การแบ่งข้อมูล (80/20 สำหรับ Validation) ---
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

    # --- [C] Parameter Tuning (ใช้แค่ Train Data) ---
    print("\n[กำลังค้นหาสถาปัตยกรรม TES ที่ดีที่สุด...]")
    best_t, best_s, best_d = grid_search_tes(train_data)
    
    print(f">>> Best Configuration:")
    print(f"    - Trend: {best_t}")
    print(f"    - Seasonal: {best_s}")
    print(f"    - Damped Trend: {best_d}")

    # --- [D] Model Training & Forecasting ---
    
    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
    # เทรนเพื่อวัดผลด้วย Train Data (80%)
    model_eval = ExponentialSmoothing(
        train_data, trend=best_t, seasonal=best_s, seasonal_periods=12, damped_trend=best_d
    ).fit(optimized=True)
    
    # ทำนายช่วง Test Data
    test_pred_tes = model_eval.forecast(len(test_data))
    rmse, wape = calculate_metrics(test_data, test_pred_tes)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (100% Data) ---")
    # เทรนโมเดลตัวจริงด้วยข้อมูลทั้งหมด (100%) เพื่อทำนายอนาคต
    final_model = ExponentialSmoothing(
        series, trend=best_t, seasonal=best_s, seasonal_periods=12, damped_trend=best_d
    ).fit(optimized=True)
    
    # ทำนายล่วงหน้า 5 ปี
    forecast_tes = final_model.forecast(forecast_months)

    # --- [E] การพล็อตแสดงผล ---
    plt.figure(figsize=(12, 6))
    
    # ข้อมูลจริง
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data (2015-2024)', linewidth=1.5)
    
    # เส้นพยากรณ์อนาคต
    conn_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    conn_val = np.concatenate([[series.values[-1]], forecast_tes.values])
    
    plt.plot(conn_idx, conn_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'TES Forecast (Next 5 years)', linewidth=1.5)

    plt.title(f'พยากรณ์อัตราการดื้อยา: {target_drug_name}\n(Holt-Winters / Triple ES)', fontsize=13, pad=15)
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

file_path = os.path.join("MDR", "model", "acinetobacter_baumannii.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    
    # กลับมาใช้การเติม 0 ตามที่คุณต้องการ
    final_df = final_df.fillna(0)
    final_df.index = all_months

    target_drug = 'AMINOGLYCOSIDES, CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        run_mdr_forecasting_tes(final_df[target_drug], "Acinetobacter baumannii")
    else:
        print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")