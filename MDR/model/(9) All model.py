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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV # <--- อัปเดตเพิ่ม RandomizedSearchCV

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
    """[อัปเดต] ค้นหาพารามิเตอร์ TES ด้วย AICc"""
    trend_opts = ['add']
    seasonal_opts = ['add', None] # ป้องกัน Error เวลาข้อมูลเป็น 0
    damped_opts = [True, False]
    
    best_aicc = float('inf')
    best_config = None
    
    for t, s, d in itertools.product(trend_opts, seasonal_opts, damped_opts):
        try:
            sp = 12 if s is not None else None
            model = ExponentialSmoothing(train_data, trend=t, seasonal=s, seasonal_periods=sp, damped_trend=d, initialization_method="estimated").fit(optimized=True)
            current_metric = getattr(model, 'aicc', model.aic)
            if current_metric < best_aicc:
                best_aicc = current_metric
                best_config = (t, s, d)
        except:
            continue
    return best_config if best_config else ('add', 'add', False)

def grid_search_ses(train_data):
    """[เพิ่มใหม่] ค้นหาพารามิเตอร์ Alpha และ Init ของ SES ด้วย AICc"""
    best_aicc = float('inf')
    best_alpha, best_init = None, None
    alphas = np.arange(0.01, 1.0, 0.05)
    
    for init in ['estimated', 'heuristic']:
        for alpha in alphas:
            try:
                model = SimpleExpSmoothing(train_data, initialization_method=init).fit(smoothing_level=alpha, optimized=False)
                current_aicc = getattr(model, 'aicc', model.aic)
                if current_aicc < best_aicc:
                    best_aicc, best_alpha, best_init = current_aicc, alpha, init
            except: continue
        try:
            model_opt = SimpleExpSmoothing(train_data, initialization_method=init).fit(optimized=True)
            current_aicc = getattr(model_opt, 'aicc', model_opt.aic)
            if current_aicc < best_aicc:
                best_aicc, best_alpha, best_init = current_aicc, model_opt.params['smoothing_level'], init
        except: continue
        
    return best_alpha if best_alpha else 0.5, best_init if best_init else 'estimated'

# ==========================================
# 2. ฟังก์ชันหลัก: ศึกประชัน 5 โมเดล (Full Option)
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
    # 1. SARIMA (Full Grid Search + AICc)
    print("[1/5] Training SARIMA (Full Grid Search)...")
    sarima_auto = auto_arima(
        train_data_std, start_p=0, start_q=0, max_p=5, max_q=5,
        start_P=0, start_Q=0, max_P=3, max_Q=3, m=12,
        seasonal=True, d=None, max_d=2, D=None, max_D=1,
        trace=False, error_action='ignore', suppress_warnings=True,
        stepwise=False, n_jobs=-1, information_criterion='aicc'
    )
    sarima_eval = SARIMAX(train_data_std, order=sarima_auto.order, seasonal_order=sarima_auto.seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred_sarima = sarima_eval.forecast(steps=len(test_data_std))
    rmse, wape = calculate_metrics(test_data_std, pred_sarima)
    results.append({'Model': 'SARIMA', 'RMSE': rmse, 'WAPE (%)': wape})
    predictions['SARIMA'] = pred_sarima

    # ------------------------------------------------
    # 2. ARIMA (Full Grid Search + AICc)
    print("[2/5] Training ARIMA (Full Grid Search)...")
    arima_auto = auto_arima(
        train_data_std, start_p=0, start_q=0, max_p=5, max_q=5,
        seasonal=False, d=None, max_d=2, test='adf',
        stepwise=False, suppress_warnings=True, error_action='ignore',
        trace=False, n_jobs=-1, information_criterion='aicc'
    )
    arima_eval = ARIMA(train_data_std, order=arima_auto.order).fit()
    pred_arima = arima_eval.forecast(steps=len(test_data_std))
    rmse, wape = calculate_metrics(test_data_std, pred_arima)
    results.append({'Model': 'ARIMA', 'RMSE': rmse, 'WAPE (%)': wape})
    predictions['ARIMA'] = pred_arima

    # ------------------------------------------------
    # 3. TES (Holt-Winters - Custom AICc Search)
    print("[3/5] Training TES (Holt-Winters)...")
    best_t, best_s, best_d = grid_search_tes(train_data_std)
    sp = 12 if best_s is not None else None
    tes_eval = ExponentialSmoothing(train_data_std, trend=best_t, seasonal=best_s, seasonal_periods=sp, damped_trend=best_d, initialization_method="estimated").fit(optimized=True)
    pred_tes = tes_eval.forecast(len(test_data_std))
    rmse, wape = calculate_metrics(test_data_std, pred_tes)
    results.append({'Model': 'TES', 'RMSE': rmse, 'WAPE (%)': wape})
    predictions['TES'] = pred_tes

    # ------------------------------------------------
    # 4. SES (Simple Exponential Smoothing - Custom Search)
    print("[4/5] Training SES...")
    best_alpha, best_init = grid_search_ses(train_data_std)
    ses_eval = SimpleExpSmoothing(train_data_std, initialization_method=best_init).fit(smoothing_level=best_alpha, optimized=False)
    pred_ses = ses_eval.forecast(len(test_data_std))
    rmse, wape = calculate_metrics(test_data_std, pred_ses)
    results.append({'Model': 'SES', 'RMSE': rmse, 'WAPE (%)': wape})
    predictions['SES'] = pred_ses

    # ------------------------------------------------
    # 5. XGBoost (RandomizedSearchCV + Regularization)
    print("[5/5] Training XGBoost (RandomizedSearchCV)...")
    X, y = create_xgb_features(series, lags=12)
    n_xgb = len(X)
    train_size_xgb = int(n_xgb * 0.80)
    X_train, y_train = X.iloc[:train_size_xgb], y.iloc[:train_size_xgb]
    X_test, y_test = X.iloc[train_size_xgb:], y.iloc[train_size_xgb:]

    tscv = TimeSeriesSplit(n_splits=3)
    param_distributions = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.5, 1],
        'reg_alpha': [0, 0.1, 1, 5],
        'reg_lambda': [0.1, 1, 5, 10]
    }
    
    xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_base, param_distributions=param_distributions, 
        n_iter=50, cv=tscv, scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=0, random_state=42
    )
    random_search.fit(X_train, y_train)
    best_params_xgb = random_search.best_params_

    xgb_eval = XGBRegressor(**best_params_xgb, objective='reg:squarederror', random_state=42)
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

file_path = os.path.join("MDR", "model","All Data", "staphylococcus_aureus.csv") 

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

    target_drug = 'AMINOGLYCOSIDES, FLUOROQUINOLONES, LINCOSAMIDES, MACROLIDES, PENICILLINS'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        run_model_showdown(series_data, "Staphylococcus aureus")
    else:
        print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")