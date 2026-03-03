import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ปิดแจ้งเตือนเพื่อความสะอาดของ Output
warnings.filterwarnings("ignore")

# ==========================================
# 1. ฟังก์ชันคำนวณ Metrics และเตรียมข้อมูล Features
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    return round(rmse, 4), round(wape, 4)

def create_features(series, lags=12):
    """สร้างตาราง Features จากข้อมูลย้อนหลัง (Lags 1-12) และเพิ่มฟีเจอร์เดือน"""
    df = pd.DataFrame(series)
    col_name = df.columns[0]
    
    # สร้าง Lag features (ประวัติย้อนหลัง 1-12 เดือน)
    for l in range(1, lags + 1):
        df[f'lag_{l}'] = df[col_name].shift(l)
    
    # เพิ่มฟีเจอร์ด้านเวลาพื้นฐาน (Month 1-12)
    df['month_num'] = df.index.month
    
    # ตัดแถวที่มี NaN ทิ้ง (12 เดือนแรกของข้อมูลจะหายไปเพราะต้องทำ Lags)
    df = df.dropna()
    X = df.drop(columns=[col_name])
    y = df[col_name]
    return X, y

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์ ทำจูน และทำนาย (XGBoost)
# ==========================================

def run_mdr_forecasting_xgb(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}")
    print(f"Analyzing (XGBoost): {target_drug_name}")
    print(f"{'='*50}")

    # --- [A] การเตรียม Features และแบ่งข้อมูล (80/20) ---
    X, y = create_features(series, lags=12)
    
    n = len(X)
    train_size = int(n * 0.80)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    # --- [B] ขั้นตอน Parameter Tuning (ใช้แค่ Train Data) ---
    print("\n[กำลังค้นหา Hyperparameters ที่ดีที่สุดด้วย TimeSeriesSplit...]")
    
    tscv = TimeSeriesSplit(n_splits=3) 
    
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f">>> Best Parameters: {best_params}")

    # --- [C] Model Training & Forecasting ---
    
    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
    model_eval = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    model_eval.fit(X_train, y_train)
    
    test_pred = model_eval.predict(X_test)
    rmse, wape = calculate_metrics(y_test, test_pred)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (100% Data) ---")
    model_final = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    model_final.fit(X, y)

    # --- [D] การพยากรณ์อนาคต 5 ปี (Recursive Forecasting) ---
    last_window = series.values[-12:].tolist() 
    future_forecast = []
    
    curr_date = series.index[-1]
    forecast_dates = pd.date_range(start=curr_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

    for i in range(forecast_months):
        feature_values = last_window[::-1] + [forecast_dates[i].month]
        input_df = pd.DataFrame([feature_values], columns=X.columns)
        pred = model_final.predict(input_df)[0]
        pred = max(0, pred) # ป้องกันค่าติดลบ
        future_forecast.append(pred)
        last_window.append(pred)
        last_window.pop(0)

    # --- [E] การพล็อตแสดงผล ---
    plt.figure(figsize=(12, 6))
    
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data (Interpolated)', linewidth=1.5)
    
    conn_idx = pd.to_datetime([series.index[-1]] + list(forecast_dates))
    conn_val = [series.values[-1]] + future_forecast
    
    plt.plot(conn_idx, conn_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label='Tuned XGBoost Forecast', linewidth=1.5)

    plt.title(f'พยากรณ์อัตราการดื้อยา: {target_drug_name}\n(XGBoost with Interpolation)', fontsize=13, pad=15)
    plt.xlabel('Year')
    plt.ylabel('Resistance Percentage (%R)')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. ส่วนการโหลดข้อมูลและการจัดการ Missing Values
# ==========================================

file_path = os.path.join("MDR", "model","All Data", "acinetobacter_baumannii.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
    # Merge เพื่อหาช่องว่างในข้อมูล
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    final_df.index = all_months

    # --- [จุดที่แก้ไข]: เปลี่ยนจาก .fillna(0) เป็น .interpolate() ---
    # ลบคอลัมน์ year/month ออกก่อนทำ interpolate เพื่อให้เหลือเฉพาะคอลัมน์เปอร์เซ็นต์
    final_df = final_df.drop(columns=['year', 'month'])
    
    # ใช้ Linear Interpolation เติมค่าว่างตามแนวโน้มจุดก่อนหน้าและถัดไป
    final_df = final_df.interpolate(method='linear')
    
    # ใช้ bfill และ ffill สำหรับข้อมูลที่ว่างที่หัวหรือท้ายตาราง (ซึ่ง interpolate ทำไม่ได้)
    final_df = final_df.bfill().ffill()

    target_drug = 'AMINOGLYCOSIDES, CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        run_mdr_forecasting_xgb(series_data, "Acinetobacter baumannii")
    else:
        print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")