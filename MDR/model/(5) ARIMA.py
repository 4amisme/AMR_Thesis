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
    sum_true = np.sum(y_true)
    wape = (np.sum(np.abs(y_true - y_pred)) / sum_true * 100) if sum_true != 0 else 0
    return round(rmse, 4), round(wape, 4)

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์และทำนาย (ARIMA)
# ==========================================

def run_mdr_forecasting_arima(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}")
    print(f"Analyzing (ARIMA): {target_drug_name}")
    print(f"{'='*50}")

    # --- [A] การแบ่งข้อมูล (80/20) ---
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

    # --- [C] Auto-ARIMA Parameter Tuning (อัปเดตใหม่) ---
    print("\n[กำลังค้นหาค่า Parameter ที่ดีที่สุด (Full Grid Search)...]")
    
    # ปรับจูน auto_arima เพื่อลด RMSE/WAPE สำหรับข้อมูลขนาดเล็ก
    stepwise_model = auto_arima(
        train_data, 
        start_p=0, start_q=0,
        max_p=5, max_q=5,              # ขยาย max_q ให้ค้นหาได้กว้างขึ้น
        d=None, max_d=2,               # ปล่อยให้ระบบหาค่า d โดยจำกัดไม่ให้เกิน 2
        test='adf',                    # ใช้ Augmented Dickey-Fuller test เพื่อหาค่า d ที่เหมาะสม
        seasonal=False,                # ไม่ใช้ Seasonality (ARIMA ปกติ)
        stepwise=False,                # ปิด Stepwise เพื่อค้นหาทุกรูปแบบ
        information_criterion='aicc',  # [สำคัญ] ใช้ AICc เหมาะกับข้อมูล Time Series ที่มีแถวน้อย
        n_jobs=-1,                     # ดึงพลัง CPU ทั้งหมดมาช่วยรันให้เร็วขึ้น
        suppress_warnings=True, 
        error_action='ignore', 
        trace=True
    )

    best_order = stepwise_model.order
    print(f"\n>>> Best ARIMA Order: {best_order}")
    print(f">>> Best AICc: {stepwise_model.aic():.2f}")

    # --- [D] Model Training & Forecasting ---
    
    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
    model_eval = ARIMA(train_data, order=best_order).fit()
    test_pred_arima = model_eval.forecast(steps=len(test_data))
    rmse, wape = calculate_metrics(test_data, test_pred_arima)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (100% Data) ---")
    final_model = ARIMA(series, order=best_order).fit()
    forecast_arima = final_model.forecast(steps=forecast_months)

    # --- [NEW] Plotting Residual Diagnostics ---
    print("\n--- 3. Plotting Residual Diagnostics ---")
    fig_diag = final_model.plot_diagnostics(figsize=(15, 8))
    fig_diag.suptitle(f'Residual Diagnostics (ARIMA): {target_drug_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # --- [E] การพล็อตแสดงผล ---
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data (Interpolated)', linewidth=1.5)
    
    forecast_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    forecast_val = np.concatenate([[series.values[-1]], forecast_arima.values])
    
    plt.plot(forecast_idx, forecast_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'Forecast (Next 5 years)', linewidth=1.5)

    plt.title(f'{target_drug_name} Multidrug-Resistant Forecast', 
              fontsize=14, fontweight='bold', pad=30) 
    plt.text(0.5, 1.03, f'Model: ARIMA | Evaluation: (RMSE: {rmse:.2f}, WAPE: {wape:.2f}%)', 
             fontsize=11, ha='center', va='bottom', transform=plt.gca().transAxes)
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

file_path = os.path.join("MDR", "model","By ward type", "s_aureus_in.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    # 1. เตรียมข้อมูล Wide Format
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    
    # สร้าง Index วันที่ให้สมบูรณ์
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
    # Merge ข้อมูล
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    final_df.index = all_months

    # ลบคอลัมน์ year, month ออกเพื่อให้เหลือแค่ค่าตัวเลขที่ต้องการ interpolate
    final_df = final_df.drop(columns=['year', 'month'])
    
    # ทำ Linear Interpolation เพื่อเติมค่าระหว่างจุด
    final_df = final_df.interpolate(method='linear')
    
    # ใช้ bfill และ ffill ในกรณีที่ค่าหัวตารางหรือท้ายตารางว่าง (ซึ่ง interpolate ทำไม่ได้)
    final_df = final_df.bfill().ffill()

    target_drug = 'LINCOSAMIDES, MACROLIDES, PENICILLINS, TETRACYCLINES'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        # เปลี่ยนชื่อใน Title กราฟจาก Pseudomonas aeruginosa เป็น Klebsiella pneumoniae ให้ตรงกับไฟล์
        run_mdr_forecasting_arima(series_data, "Staphylococcus aureus")
    else:
        print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")