import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV 

import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf

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
    print("\n[กำลังค้นหา Hyperparameters ที่ดีที่สุดด้วย RandomizedSearchCV...]")
    
    tscv = TimeSeriesSplit(n_splits=3) 
    
    param_distributions = {
        'n_estimators': [100, 300, 500, 800, 1000],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.5, 1, 2],
        'reg_alpha': [0, 0.1, 1, 5, 10],
        'reg_lambda': [0.1, 1, 5, 10, 50]
    }

    xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # ใช้ RandomizedSearchCV เพื่อให้รันได้เร็วและครอบคลุม
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=100,           # สุ่มหา 100 รูปแบบ
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print(f">>> Best Parameters: {best_params}")

    # --- [C] Model Training & Forecasting ---
    
    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
    model_eval = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    model_eval.fit(X_train, y_train)
    
    # [UPDATE] ล็อกค่าพยากรณ์ไม่ให้เกิน 0-100%
    test_pred = np.clip(model_eval.predict(X_test), 0, 100)
    rmse, wape = calculate_metrics(y_test, test_pred)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (100% Data) ---")
    model_final = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    model_final.fit(X, y)

    # --- 3. Plotting Residual Diagnostics ---
    print("\n--- 3. Plotting Residual Diagnostics ---")
    # [UPDATE] ล็อกค่าพยากรณ์ไม่ให้เกิน 0-100% สำหรับการคำนวณ Residuals
    y_pred_all = np.clip(model_final.predict(X), 0, 100)
    residuals = y - y_pred_all
    
    fig_diag, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig_diag.suptitle(f'Residual Diagnostics (XGBoost): {target_drug_name}', fontsize=14, y=1.02)
    
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

    # --- [D] การพยากรณ์อนาคต (Recursive Forecasting) ---
    last_window = series.values[-12:].tolist() 
    future_forecast = []
    
    curr_date = series.index[-1]
    forecast_dates = pd.date_range(start=curr_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

    for i in range(forecast_months):
        feature_values = last_window[::-1] + [forecast_dates[i].month]
        input_df = pd.DataFrame([feature_values], columns=X.columns)
        
        # [UPDATE] ล็อกค่าให้อยู่ระหว่าง 0 ถึง 100% เท่านั้น
        pred = np.clip(model_final.predict(input_df)[0], 0, 100)
        
        future_forecast.append(pred)
        last_window.append(pred)
        last_window.pop(0)

    # --- [E] การพล็อตแสดงผลพยากรณ์อนาคต ---
    plt.figure(figsize=(12, 6))
    
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data (Interpolated)', linewidth=1.5)
    
    conn_idx = pd.to_datetime([series.index[-1]] + list(forecast_dates))
    conn_val = [series.values[-1]] + future_forecast
    
    plt.plot(conn_idx, conn_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'Forecast (Next {forecast_months//12} years)', linewidth=1.5)
    
    plt.title(f'{target_drug_name} Forecast', fontsize=14, fontweight='bold', pad=30) 
    plt.text(0.5, 1.03, f'Model: XGBoost | Evaluation: (RMSE: {rmse:.2f}, WAPE: {wape:.2f}%)', 
             fontsize=11, ha='center', va='bottom', transform=plt.gca().transAxes)
    
    plt.xlabel('Year')
    plt.ylabel('Resistance Percentage (%R)')
    plt.ylim(0, max(100, series.max() + 5)) # กำหนดสเกลแกน Y ให้ครอบคลุมอย่างน้อย 100%
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. ส่วนการโหลดข้อมูลและการจัดการ Missing Values
# ==========================================

file_path = os.path.join("MDR", "model_for_1_Drug", "Ward", "e_coli_out.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    # [UPDATE] เปลี่ยนเป็น resistant_drug_name และใช้ aggfunc='mean'
    pivot_df = df.pivot_table(
        index=['year', 'month'], 
        columns='resistant_drug_name', 
        values='percentage',
        aggfunc='mean'
    )
    
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
    # Merge เพื่อหาช่องว่างในข้อมูล
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    final_df.index = all_months

    # ลบคอลัมน์ year/month ออกก่อนทำ interpolate
    final_df = final_df.drop(columns=['year', 'month'])
    
    # ใช้ Linear Interpolation และ bfill/ffill จัดการค่าว่าง
    final_df = final_df.interpolate(method='linear')
    final_df = final_df.bfill().ffill()

    # [UPDATE] เปลี่ยนเป็นชื่อยาที่คุณมีใน Dataset
    target_drug = 'cefuroxime' # สามารถแก้เป็นชื่อยาอื่นได้ เช่น 'Oxacillin', 'Vancomycin'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        # [UPDATE] ปรับชื่อ Title ให้ตรงกับสายพันธุ์ P. aeruginosa
        run_mdr_forecasting_xgb(series_data, f"Escherichia coli to {target_drug}")
    else:
        print(f"❌ ไม่พบชื่อยา '{target_drug}' ในข้อมูล")
        print(f"รายชื่อยาที่มีทั้งหมด: {list(final_df.columns)}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")