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
# 1. ฟังก์ชันคำนวณ Metrics และ Tuning
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE สำหรับวัดประสิทธิภาพ"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    return round(rmse, 4), round(wape, 4)

def tune_ses_alpha(train, val):
    """ทำ Grid Search เพื่อหาค่า Alpha ที่ดีที่สุดบน Validation Set"""
    best_alpha = None
    best_rmse = float('inf')
    
    # ทดลองค่า alpha ตั้งแต่ 0.01 ถึง 0.99 โดยละเอียด
    alphas = np.linspace(0.01, 0.99, 50)
    
    for a in alphas:
        try:
            # เทรนโมเดลด้วยค่า alpha ที่กำหนด
            model = SimpleExpSmoothing(train, initialization_method="estimated").fit(
                smoothing_level=a, 
                optimized=False
            )
            # พยากรณ์บน Validation Set
            pred = model.forecast(len(val))
            rmse = np.sqrt(mean_squared_error(val, pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = a
        except:
            continue
            
    return round(best_alpha, 4), round(best_rmse, 4)

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์และทำนาย
# ==========================================

def run_mdr_forecasting_ses_tuned(series, target_drug_name, forecast_months=60):
    # --- [A] การแบ่งข้อมูล (80/10/10) ---
    n = len(series)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    
    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]

    # --- [B] Parameter Tuning (Grid Search) ---
    print(f"กำลังค้นหาค่า Alpha ที่เหมาะสมที่สุดสำหรับ {target_drug_name}...")
    best_alpha, val_rmse = tune_ses_alpha(train, val)
    print(f"Best Alpha found: {best_alpha} (Validation RMSE: {val_rmse})")

    # --- [C] Model Training (ใช้ Best Alpha เทรนข้อมูลทั้งหมดก่อน Test) ---
    train_val_combined = series.iloc[:val_end]
    
    # เทรนโมเดลสุดท้ายด้วยค่า Alpha ที่ดีที่สุด
    model_final = SimpleExpSmoothing(train_val_combined, initialization_method="estimated").fit(
        smoothing_level=best_alpha, 
        optimized=False
    )
    
    # วัดผลประสิทธิภาพจริงด้วย Test Set (ข้อมูลปี 2024)
    test_pred = model_final.forecast(len(test))
    rmse, wape = calculate_metrics(test, test_pred)
    
    # ดึงค่า AIC เพื่อใช้ประกอบรายงาน
    aic_value = round(model_final.aic, 4)
    
    print("-" * 50)
    print(f"Tuned Simple ES Result (Test Set 2024):")
    print(f"Best Alpha: {best_alpha}")
    print(f"AIC: {aic_value}")
    print(f"RMSE: {rmse}")
    print(f"WAPE: {wape}%")
    print("-" * 50)

    # --- [D] การพยากรณ์อนาคต 5 ปี ---
    forecast_future = model_final.forecast(steps=forecast_months)

    # --- [E] การพล็อตแสดงผล ---
    plt.figure(figsize=(12, 6))
    
    # ข้อมูลจริง
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data', linewidth=1.5)
    
    # เส้นพยากรณ์ (ลากจากจุดสุดท้ายของข้อมูลจริง)
    connection_idx = pd.to_datetime([series.index[-1]] + list(forecast_future.index))
    connection_val = [series.values[-1]] + list(forecast_future.values)
    
    plt.plot(connection_idx, connection_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'SES Forecast (Alpha={best_alpha})', linewidth=1.5)

    plt.title(f'Simple Exponential Smoothing (Tuned): {target_drug_name}', fontsize=13, pad=15)
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

    # ระบุกลุ่มยาเป้าหมาย
    target_drug = 'CARBAPENEMS, CEPHEMS, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        run_mdr_forecasting_ses_tuned(final_df[target_drug], "Acinetobacter baumannii")
    else:
        print(f"ไม่พบกลุ่มยา: {target_drug}")