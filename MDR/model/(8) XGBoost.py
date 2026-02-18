import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ==========================================
# 1. ฟังก์ชันคำนวณ Metrics และเตรียมข้อมูล Features
# ==========================================

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # คำนวณ WAPE (Weighted Average Relative Error)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    return round(rmse, 4), round(wape, 4)

def create_features(series, lags=12):
    """สร้างตาราง Features จากข้อมูลย้อนหลัง (Lags)"""
    df = pd.DataFrame(series)
    col_name = df.columns[0]
    
    # สร้าง Lag features (เดือนที่ 1-12 ย้อนหลัง)
    for l in range(1, lags + 1):
        df[f'lag_{l}'] = df[col_name].shift(l)
    
    # เพิ่มฟีเจอร์ด้านเวลาพื้นฐาน
    df['month_num'] = df.index.month
    
    df = df.dropna()
    X = df.drop(columns=[col_name])
    y = df[col_name]
    return X, y

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์ ทำจูน และทำนาย
# ==========================================

def run_mdr_forecasting_xgb_tuned(series, target_drug_name, forecast_months=60):
    # --- [A] การเตรียม Features และแบ่งข้อมูล (80/10/10) ---
    X, y = create_features(series, lags=12)
    
    n = len(X)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    
    X_train_full, y_train_full = X.iloc[:val_end], y.iloc[:val_end] # สำหรับ Tuning
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]            # สำหรับ Final Test

    # --- [B] ขั้นตอน Parameter Tuning (Grid Search + TimeSeriesSplit) ---
    print(f"Starting Hyperparameter Tuning for {target_drug_name}...")
    
    # ใช้ TimeSeriesSplit เพื่อป้องกัน Data Leakage (ไม่สุ่มข้อมูล)
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }

    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_full, y_train_full)
    best_params = grid_search.best_params_
    
    print(f"Best Parameters found: {best_params}")

    # --- [C] เทรนโมเดลที่ดีที่สุดพร้อม Early Stopping ---
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

    # ใส่ early_stopping_rounds ใน Constructor ของ XGBRegressor
    final_model = XGBRegressor(
        **best_params, 
        objective='reg:squarederror', 
        random_state=42,
        early_stopping_rounds=50
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # --- [D] การวัดผลด้วย Test Set ---
    test_pred = final_model.predict(X_test)
    rmse, wape = calculate_metrics(y_test, test_pred)
    
    print("-" * 50)
    print(f"Tuned XGBoost Evaluation Result:")
    print(f"RMSE: {rmse}")
    print(f"WAPE: {wape}%")
    print("-" * 50)

    # --- [E] การพยากรณ์อนาคต 5 ปี (Recursive Forecasting) ---
    last_window = series.values[-12:].tolist() 
    future_forecast = []
    
    curr_date = series.index[-1]
    forecast_dates = pd.date_range(start=curr_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

    for i in range(forecast_months):
        # สร้าง Feature สำหรับเดือนที่จะทาย (Lags 1-12 เรียงจากใหม่ไปเก่า + Month number)
        input_features = np.array(last_window[::-1] + [forecast_dates[i].month]).reshape(1, -1)
        
        # แปลงเป็น DataFrame เพื่อให้มีชื่อ Column เหมือนตอน Train (ป้องกัน Warning)
        input_df = pd.DataFrame(input_features, columns=X_train.columns)
        pred = final_model.predict(input_df)[0]
        
        # ป้องกันค่าติดลบ (ถ้ามี)
        pred = max(0, pred)
        
        future_forecast.append(pred)
        last_window.append(pred)
        last_window.pop(0)

    # --- [F] การพล็อตแสดงผล ---
    plt.figure(figsize=(14, 7))
    plt.plot(series.index, series.values, color='#377eb8', marker='o', markersize=4, label='Actual Historical Data', linewidth=1.5)
    
    # เชื่อมจุดสุดท้ายของข้อมูลจริงกับจุดแรกของพยากรณ์
    connection_idx = [series.index[-1]] + list(forecast_dates)
    connection_val = [series.values[-1]] + future_forecast
    
    plt.plot(connection_idx, connection_val, color='#e41a1c', marker='o', markersize=4, linestyle='--', label='Tuned XGBoost Forecast (5 years)', linewidth=1.5)

    plt.title(f'Tuned XGBoost Time Series Forecasting: {target_drug_name}', fontsize=14, pad=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Resistance Percentage (%R)', fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. ส่วนการรันข้อมูล
# ==========================================

# ปรับ Path ตามที่ใช้งานจริง
file_path = os.path.join("MDR", "model", "acinetobacter_baumannii.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    # จัดเตรียมข้อมูลเหมือนเดิม
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left').fillna(0)
    final_df.index = all_months

    # ระบุกลุ่มยาเป้าหมาย
    target_drug = 'CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        run_mdr_forecasting_xgb_tuned(series_data, "Acinetobacter baumannii")
    else:
        print(f"Drug class '{target_drug}' not found in data.")
else:
    print(f"File not found at {file_path}")