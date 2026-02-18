import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import itertools
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import warnings

# ปิดการแจ้งเตือน ConvergenceWarning เพื่อความสะอาดของ Output
warnings.filterwarnings("ignore")

# ==========================================
# 1. ฟังก์ชันสำหรับการ Tuning และ Metrics
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    return round(rmse, 4), round(wape, 4)

def grid_search_tes(train, val):
    """ทำ Parameter Tuning เพื่อหา Configuration ที่ดีที่สุดบน Validation Set"""
    trend_opts = ['add', None]  # แนะนำให้ใช้ add เป็นหลักสำหรับข้อมูล % 
    seasonal_opts = ['add', None] 
    damped_opts = [True, False]
    
    best_rmse = float('inf')
    best_config = None
    
    # สร้างรายการการตั้งค่าทั้งหมดที่เป็นไปได้
    configs = list(itertools.product(trend_opts, seasonal_opts, damped_opts))
    
    for t, s, d in configs:
        try:
            # เทรนบน Train และวัดผลบน Val
            model = ExponentialSmoothing(
                train, trend=t, seasonal=s, seasonal_periods=12, damped_trend=d
            ).fit(optimized=True)
            
            pred = model.forecast(len(val))
            rmse = np.sqrt(mean_squared_error(val, pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_config = (t, s, d)
        except:
            continue
            
    return best_config

# ==========================================
# 2. ฟังก์ชันหลักสำหรับการวิเคราะห์
# ==========================================

def run_mdr_forecasting_tes_tuned(series, target_drug_name, forecast_months=60):
    # --- [A] การแบ่งข้อมูล (80/10/10) ---
    n = len(series)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    
    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]

    # --- [B] Parameter Tuning (Grid Search) ---
    print(f"กำลังค้นหาพารามิเตอร์ที่เหมาะสมที่สุดสำหรับ {target_drug_name}...")
    best_t, best_s, best_d = grid_search_tes(train, val)
    print(f"Best Configuration found: Trend={best_t}, Seasonal={best_s}, Damped={best_d}")

    # --- [C] Model Training (ใช้ Best Config เทรนบนข้อมูลก่อน Test ทั้งหมด) ---
    train_val_combined = series.iloc[:val_end]
    
    model_final = ExponentialSmoothing(
        train_val_combined, 
        trend=best_t, 
        seasonal=best_s, 
        seasonal_periods=12,
        damped_trend=best_d
    ).fit()
    
    # ทำนาย Test Set และ อนาคต 5 ปี
    test_pred = model_final.forecast(len(test))
    forecast_future = model_final.forecast(forecast_months)
    
    # วัดผลประสิทธิภาพจริงด้วย Test Set (ข้อมูลปี 2024)
    rmse, wape = calculate_metrics(test, test_pred)
    
    print("-" * 50)
    print(f"Tuned Triple ES (Holt-Winters) Result:")
    print(f"RMSE (Test Set): {rmse}")
    print(f"WAPE (Test Set): {wape}%")
    print("-" * 50)

    # --- [D] การพล็อตแสดงผล ---
    plt.figure(figsize=(12, 6))
    
    # ข้อมูลจริง
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data', linewidth=1.5)
    
    # เส้นพยากรณ์
    conn_idx = pd.to_datetime([series.index[-1]] + list(forecast_future.index))
    conn_val = [series.values[-1]] + list(forecast_future.values)
    
    plt.plot(conn_idx, conn_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'Tuned TES Forecast (5 yrs)', linewidth=1.5)

    plt.title(f'Triple Exponential Smoothing (Tuned): {target_drug_name}', fontsize=13, pad=15)
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
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left').fillna(0)
    final_df.index = all_months

    target_drug = 'AMINOGLYCOSIDES, CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        run_mdr_forecasting_tes_tuned(final_df[target_drug], "Acinetobacter baumannii")