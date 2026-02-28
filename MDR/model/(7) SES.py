import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import warnings

# ปิดการแจ้งเตือนเพื่อให้ Output สะอาด
warnings.filterwarnings("ignore")

# ==========================================
# 1. ฟังก์ชันคำนวณ Metrics
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE สำหรับวัดประสิทธิภาพ"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    return round(rmse, 4), round(wape, 4)

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์และทำนาย (SES)
# ==========================================

def run_mdr_forecasting_ses(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}")
    print(f"Analyzing (Simple Exponential Smoothing): {target_drug_name}")
    print(f"{'='*50}")

    # --- [A] การแบ่งข้อมูล (80/20 สำหรับ Validation) ---
    n = len(series)
    train_size = int(n * 0.80)
    train_data = series.iloc[:train_size]
    test_data = series.iloc[train_size:]

    # --- [B] Model Training & Forecasting (กระบวนการ 2 ขั้นตอน) ---
    
    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
    # เทรนและหาค่า Alpha ที่ดีที่สุดอัตโนมัติด้วย Train Data (80%)
    model_eval = SimpleExpSmoothing(train_data, initialization_method="estimated").fit(optimized=True)
    
    best_alpha_eval = round(model_eval.params['smoothing_level'], 4)
    print(f">>> Optimized Alpha (Train): {best_alpha_eval}")
    
    # ทำนายช่วง Test Data และวัดผล
    test_pred_ses = model_eval.forecast(len(test_data))
    rmse, wape = calculate_metrics(test_data, test_pred_ses)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (100% Data) ---")
    # เทรนโมเดลตัวจริงด้วยข้อมูลทั้งหมด (100%) เพื่อหา Alpha ที่อัปเดตที่สุดก่อนทำนายอนาคต
    final_model = SimpleExpSmoothing(series, initialization_method="estimated").fit(optimized=True)
    
    best_alpha_final = round(final_model.params['smoothing_level'], 4)
    print(f">>> Optimized Alpha (100% Data): {best_alpha_final}")
    
    # ทำนายล่วงหน้า 5 ปี
    forecast_ses = final_model.forecast(forecast_months)

    # --- [C] การพล็อตแสดงผล ---
    
    plt.figure(figsize=(12, 6))
    
    # ข้อมูลจริง
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data (2015-2024)', linewidth=1.5)
    
    # เส้นพยากรณ์อนาคต
    conn_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    conn_val = np.concatenate([[series.values[-1]], forecast_ses.values])
    
    plt.plot(conn_idx, conn_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'SES Forecast (Next 5 years)', linewidth=1.5)

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
    
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    
    # ใช้การเติม 0 ตามสเปคที่ต้องการ
    final_df = final_df.fillna(0)
    final_df.index = all_months

    target_drug = 'CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        run_mdr_forecasting_ses(final_df[target_drug], "Acinetobacter baumannii")
    else:
        print(f"ไม่พบกลุ่มยา: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")