import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
import itertools

# นำเข้าเครื่องมือจากไลบรารีต่างๆ
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ปิดการแจ้งเตือน
warnings.filterwarnings("ignore")

# ==========================================
# 1. ฟังก์ชันเตรียมข้อมูลและคำนวณ Metrics
# ==========================================

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sum_true = np.sum(y_true)
    wape = (np.sum(np.abs(y_true - y_pred)) / sum_true * 100) if sum_true != 0 else 0
    return round(rmse, 4), round(wape, 4)

def create_xgb_features(series, lags=12):
    df = pd.DataFrame(series)
    col = df.columns[0]
    for l in range(1, lags + 1):
        df[f'lag_{l}'] = df[col].shift(l)
    df['month_num'] = df.index.month
    df = df.dropna()
    return df.drop(columns=[col]), df[col]

def grid_search_tes(train_data):
    trend_opts, seasonal_opts, damped_opts = ['add'], ['add'], [True, False]
    best_aic, best_config = float('inf'), None
    for t, s, d in itertools.product(trend_opts, seasonal_opts, damped_opts):
        try:
            model = ExponentialSmoothing(train_data, trend=t, seasonal=s, seasonal_periods=12, damped_trend=d).fit(optimized=True)
            if model.aic < best_aic:
                best_aic, best_config = model.aic, (t, s, d)
        except: continue
    return best_config if best_config else ('add', 'add', False)

# ==========================================
# 2. ฟังก์ชันหลัก: ศึกประชัน 5 โมเดล (โหมดเหมือนเป๊ะ)
# ==========================================

def run_model_showdown(series, target_drug_name):
    print(f"\n{'='*60}")
    print(f"🚀 MODEL SHOWDOWN (Full Option): {target_drug_name}")
    print(f"{'='*60}\n")

    # --- การแบ่งข้อมูลมาตรฐาน สำหรับ 4 โมเดลแรก ---
    n_standard = len(series)
    train_size_std = int(n_standard * 0.80)
    train_data_std = series.iloc[:train_size_std]
    test_data_std = series.iloc[train_size_std:]
    
    results = []
    predictions = {} 

    # ------------------------------------------------
    # 1. SARIMA (ขั้นตอนเดียวกับรันเดี่ยว)
    print("[1/5] Training SARIMA (Stepwise Search + Re-fit)...")
    sarima_auto = auto_arima(
        train_data_std, start_p=0, start_q=0, max_p=5, max_q=5, m=12,
        seasonal=True, d=None, D=1, trace=False, error_action='ignore',
        suppress_warnings=True, stepwise=True
    )
    sarima_eval = SARIMAX(train_data_std, order=sarima_auto.order, seasonal_order=sarima_auto.seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred_sarima = sarima_eval.forecast(steps=len(test_data_std))
    rmse, wape = calculate_metrics(test_data_std, pred_sarima)
    results.append({'Model': 'SARIMA', 'RMSE': rmse, 'WAPE (%)': wape})
    predictions['SARIMA'] = pred_sarima

    # ------------------------------------------------
    # 2. ARIMA (ขั้นตอนเดียวกับรันเดี่ยว)
    print("[2/5] Training ARIMA (Grid Search + Re-fit)...")
    arima_auto = auto_arima(
        train_data_std, start_p=0, start_q=0, max_p=5, max_q=3,
        seasonal=False, stepwise=False, suppress_warnings=True,
        error_action='ignore', trace=False
    )
    arima_eval = ARIMA(train_data_std, order=arima_auto.order).fit()
    pred_arima = arima_eval.forecast(steps=len(test_data_std))
    rmse, wape = calculate_metrics(test_data_std, pred_arima)
    results.append({'Model': 'ARIMA', 'RMSE': rmse, 'WAPE (%)': wape})
    predictions['ARIMA'] = pred_arima

    # ------------------------------------------------
    # 3. TES (Holt-Winters)
    print("[3/5] Training TES (Holt-Winters)...")
    best_t, best_s, best_d = grid_search_tes(train_data_std)
    tes_eval = ExponentialSmoothing(train_data_std, trend=best_t, seasonal=best_s, seasonal_periods=12, damped_trend=best_d).fit(optimized=True)
    pred_tes = tes_eval.forecast(len(test_data_std))
    rmse, wape = calculate_metrics(test_data_std, pred_tes)
    results.append({'Model': 'TES', 'RMSE': rmse, 'WAPE (%)': wape})
    predictions['TES'] = pred_tes

    # ------------------------------------------------
    # 4. SES (Simple Exponential Smoothing)
    print("[4/5] Training SES...")
    ses_eval = SimpleExpSmoothing(train_data_std, initialization_method="estimated").fit(optimized=True)
    pred_ses = ses_eval.forecast(len(test_data_std))
    rmse, wape = calculate_metrics(test_data_std, pred_ses)
    results.append({'Model': 'SES', 'RMSE': rmse, 'WAPE (%)': wape})
    predictions['SES'] = pred_ses

    # ------------------------------------------------
    # 5. XGBoost (การแบ่งข้อมูลแยกเฉพาะให้ตรงกับรันเดี่ยวเป๊ะๆ)
    print("[5/5] Training XGBoost (Full Grid Search)...")
    X, y = create_xgb_features(series, lags=12)
    n_xgb = len(X)
    train_size_xgb = int(n_xgb * 0.80)
    X_train, y_train = X.iloc[:train_size_xgb], y.iloc[:train_size_xgb]
    X_test, y_test = X.iloc[train_size_xgb:], y.iloc[train_size_xgb:]

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
        estimator=xgb_base, param_grid=param_grid, cv=tscv,
        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    xgb_eval = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    xgb_eval.fit(X_train, y_train)
    
    pred_xgb_raw = xgb_eval.predict(X_test)
    pred_xgb = pd.Series(pred_xgb_raw, index=X_test.index)
    rmse, wape = calculate_metrics(y_test, pred_xgb)
    results.append({'Model': 'XGBoost', 'RMSE': rmse, 'WAPE (%)': wape})
    predictions['XGBoost'] = pred_xgb

    # ==========================================
    # สรุปผลลัพธ์ (Leaderboard)
    # ==========================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='WAPE (%)', ascending=True).reset_index(drop=True)
    
    print("\n🏆 --- MODEL LEADERBOARD --- 🏆")
    # ใช้ to_string แทน to_markdown เพื่อไม่ให้ติด error tabulate
    print(results_df.to_string(index=False)) 
    
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n🌟 ผู้ชนะคือ: **{best_model_name}** (Error ต่ำที่สุด!)")

    # ==========================================
    # พล็อตกราฟเปรียบเทียบ (Test Data)
    # ==========================================
    plt.figure(figsize=(14, 7))
    
    # พล็อต Train & Test Data (ของจริง)
    plt.plot(series.index[:train_size_std], series.values[:train_size_std], color='black', label='Train Data', linewidth=1.5)
    plt.plot(series.index[train_size_std:], series.values[train_size_std:], color='black', label='Test Data (Actual)', linewidth=2.5, linestyle=':')
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for idx, (model_name, pred_series) in enumerate(predictions.items()):
        plt.plot(pred_series.index, pred_series.values, color=colors[idx], label=f'{model_name} Forecast', linewidth=1.5, alpha=0.8)

    plt.axvline(x=test_data_std.index[0], color='gray', linestyle='--', alpha=0.5, label='Train/Test Split')
    
    plt.title(f'Test Set Forecasting Comparison: {target_drug_name}', fontsize=14, pad=15)
    plt.xlabel('Year')
    plt.ylabel('Resistance Percentage (%R)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.grid(True, which='both', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. โหลดข้อมูลและรัน Showdown
# ==========================================

file_path = os.path.join("MDR", "model","By_specimen", "k_pneumoniae_ps.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    final_df.index = all_months
    final_df = final_df.drop(columns=['year', 'month'])
    
    final_df = final_df.interpolate(method='linear')
    final_df = final_df.bfill().ffill()

    target_drug = 'CEPHEMS, FLUOROQUINOLONES, FOLATE PATHWAY ANTAGONISTS, PENICILLINS, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        run_model_showdown(series_data, "Pseudomonas aeruginosa")
    else:
        print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")