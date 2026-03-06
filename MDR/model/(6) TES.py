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

<<<<<<< HEAD
# --- [NEW] เพิ่ม Import สำหรับทำ Residual Plot ---
import statsmodels.api as sm
import scipy.stats as stats

=======
>>>>>>> main
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
<<<<<<< HEAD
    trend_opts = ['add']      
    seasonal_opts = ['add']   
    damped_opts = [True, False]     
=======
    
    # 📌 จุดที่แก้ไข: บังคับให้ต้องมี Trend และ Seasonal เสมอ (ตัด None ทิ้ง)
    trend_opts = ['add']      
    seasonal_opts = ['add']   
    damped_opts = [True, False]     # หางค้าง (ลดความชันเทรนด์ในอนาคต)
>>>>>>> main
    
    best_aic = float('inf')
    best_config = None
    
<<<<<<< HEAD
=======
    # สร้างรายการการตั้งค่าทั้งหมดที่เป็นไปได้
>>>>>>> main
    configs = list(itertools.product(trend_opts, seasonal_opts, damped_opts))
    
    for t, s, d in configs:
        try:
<<<<<<< HEAD
=======
            # เทรนและหาค่า AIC บนชุดข้อมูล Train
>>>>>>> main
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

<<<<<<< HEAD
    # --- [A] การแบ่งข้อมูล (80/20) ---
=======
    # --- [A] การแบ่งข้อมูล (80/20 สำหรับ Validation) ---
>>>>>>> main
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

<<<<<<< HEAD
    # --- [C] Parameter Tuning ---
=======
    # --- [C] Parameter Tuning (ใช้แค่ Train Data) ---
>>>>>>> main
    print("\n[กำลังค้นหาสถาปัตยกรรม TES ที่ดีที่สุด...]")
    best_t, best_s, best_d = grid_search_tes(train_data)
    
    print(f">>> Best Configuration:")
    print(f"    - Trend: {best_t}")
    print(f"    - Seasonal: {best_s}")
    print(f"    - Damped Trend: {best_d}")

    # --- [D] Model Training & Forecasting ---
    
    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
<<<<<<< HEAD
=======
    # เทรนเพื่อวัดผลด้วย Train Data (80%)
>>>>>>> main
    model_eval = ExponentialSmoothing(
        train_data, trend=best_t, seasonal=best_s, seasonal_periods=12, damped_trend=best_d
    ).fit(optimized=True)
    
<<<<<<< HEAD
=======
    # ทำนายช่วง Test Data
>>>>>>> main
    test_pred_tes = model_eval.forecast(len(test_data))
    rmse, wape = calculate_metrics(test_data, test_pred_tes)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (100% Data) ---")
<<<<<<< HEAD
=======
    # เทรนโมเดลตัวจริงด้วยข้อมูลทั้งหมด (100%) เพื่อทำนายอนาคต
>>>>>>> main
    final_model = ExponentialSmoothing(
        series, trend=best_t, seasonal=best_s, seasonal_periods=12, damped_trend=best_d
    ).fit(optimized=True)
    
<<<<<<< HEAD
    forecast_tes = final_model.forecast(forecast_months)

    # --- [NEW] Plotting Residual Diagnostics (Manual for TES) ---
    print("\n--- 3. Plotting Residual Diagnostics ---")
    residuals = final_model.resid
    
    fig_diag, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig_diag.suptitle(f'Residual Diagnostics (TES): {target_drug_name}', fontsize=14, y=1.02)
    
    # 1. Standardized residual (Top Left)
    axes[0, 0].plot(residuals.index, residuals.values)
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Residuals over time')
    
    # 2. Histogram plus estimated density (Top Right)
    axes[0, 1].hist(residuals, density=True, bins=15, color='#377eb8', edgecolor='white', label='Hist')
    kde = stats.gaussian_kde(residuals.dropna())
    x_kde = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x_kde, kde(x_kde), color='#ff7f00', label='KDE')
    mu, std = stats.norm.fit(residuals.dropna())
    p = stats.norm.pdf(x_kde, mu, std)
    axes[0, 1].plot(x_kde, p, color='#4daf4a', label='N(0,1)')
    axes[0, 1].set_title('Histogram plus estimated density')
    axes[0, 1].legend()
    
    # 3. Normal Q-Q (Bottom Left)
    sm.qqplot(residuals.dropna(), line='s', ax=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q')
    
    # 4. Correlogram (Bottom Right)
    plot_acf(residuals.dropna(), lags=24, ax=axes[1, 1])
    axes[1, 1].set_title('Correlogram')
    
    plt.tight_layout()
    plt.show()

    # --- [E] การพล็อตแสดงผลพยากรณ์ ---
=======
    # ทำนายล่วงหน้า 5 ปี
    forecast_tes = final_model.forecast(forecast_months)

    # --- [E] การพล็อตแสดงผล ---
    # [Image of Triple Exponential Smoothing forecast with seasonality and trend]
>>>>>>> main
    plt.figure(figsize=(12, 6))
    
    # ข้อมูลจริง
    plt.plot(series.index, series.values, 
<<<<<<< HEAD
             color='#377eb8', marker='o', markersize=4, label='Actual Data (Interpolated)', linewidth=1.5)
=======
             color='#377eb8', marker='o', markersize=4, label='Actual Data (2015-2024)', linewidth=1.5)
>>>>>>> main
    
    # เส้นพยากรณ์อนาคต
    conn_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    conn_val = np.concatenate([[series.values[-1]], forecast_tes.values])
    
    plt.plot(conn_idx, conn_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'TES Forecast (Next 5 years)', linewidth=1.5)

<<<<<<< HEAD
    plt.title(f'MDR Pattern Prediction: {target_drug_name}\n(TES with Linear Interpolation)', fontsize=13, pad=15)
=======
    plt.title(f'MDR Pattern Prediction: {target_drug_name}\n(Forced Triple ES)', fontsize=13, pad=15)
>>>>>>> main
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

<<<<<<< HEAD
file_path = os.path.join("MDR", "model","All Data", "acinetobacter_baumannii.csv") 
=======
file_path = os.path.join("MDR", "model", "a_baumannii_ur.csv") 
>>>>>>> main

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
<<<<<<< HEAD
    # Merge ข้อมูลเข้ากับช่วงเวลาทั้งหมด
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    
    # ลบคอลัมน์ year และ month ออกก่อนทำ interpolate เพื่อให้เหลือเฉพาะค่าตัวเลขที่ต้องการ
    final_df.index = all_months
    final_df = final_df.drop(columns=['year', 'month'])
    
    # ใช้ Linear Interpolation เพื่อเติมค่าที่ขาดหายไปตามแนวโน้ม
    final_df = final_df.interpolate(method='linear')
    
    # เก็บตกกรณีค่าว่างที่หัวหรือท้ายตารางที่ interpolate เข้าไม่ถึง
    final_df = final_df.bfill().ffill()

    target_drug = 'AMINOGLYCOSIDES, CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS'
=======
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    
    # ใช้การเติม 0 ตามที่คุณต้องการ
    final_df = final_df.fillna(0)
    final_df.index = all_months

    target_drug = 'CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, β-LACTAM COMBINATION AGENTS'
>>>>>>> main

    if target_drug in final_df.columns:
        run_mdr_forecasting_tes(final_df[target_drug], "Acinetobacter baumannii")
    else:
        print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")