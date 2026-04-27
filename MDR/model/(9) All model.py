import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
import itertools

# โมเดลและเครื่องมือทางสถิติ
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV 
from sklearn.metrics import mean_squared_error

# เครื่องมือสำหรับวาดกราฟ Residual
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ปิดแจ้งเตือน Warning เพื่อความสะอาดของ Output
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# ==========================================
# 1. ฟังก์ชันส่วนกลาง (Shared Functions)
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE สำหรับวัดประสิทธิภาพ"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sum_true = np.sum(y_true)
    wape = (np.sum(np.abs(y_true - y_pred)) / sum_true * 100) if sum_true != 0 else 0
    return round(rmse, 4), round(wape, 4)

def grid_search_tes(train_data):
    trend_opts = ['add']           
    seasonal_opts = ['add', None]  
    damped_opts = [True, False]     
    best_aicc = float('inf')
    best_config = None
    
    for t in trend_opts:
        for s in seasonal_opts:
            for d in damped_opts:
                try:
                    sp = 12 if s is not None else None
                    model = ExponentialSmoothing(
                        train_data, trend=t, seasonal=s, seasonal_periods=sp, damped_trend=d, initialization_method="estimated"
                    ).fit(optimized=True)
                    current_metric = getattr(model, 'aicc', model.aic)
                    if current_metric < best_aicc:
                        best_aicc = current_metric
                        best_config = (t, s, d)
                except:
                    continue
    if best_config is None:
        return ('add', 'add', False) 
    return best_config

def grid_search_ses(train_data):
    best_aicc = float('inf')
    best_alpha = None
    best_init = None
    alphas = np.arange(0.01, 1.0, 0.05)
    init_methods = ['estimated', 'heuristic']
    
    for init in init_methods:
        for alpha in alphas:
            try:
                model = SimpleExpSmoothing(train_data, initialization_method=init).fit(smoothing_level=alpha, optimized=False)
                current_aicc = getattr(model, 'aicc', model.aic)
                if current_aicc < best_aicc:
                    best_aicc = current_aicc
                    best_alpha = alpha
                    best_init = init
            except:
                continue
        try:
            model_opt = SimpleExpSmoothing(train_data, initialization_method=init).fit(optimized=True)
            current_aicc = getattr(model_opt, 'aicc', model_opt.aic)
            if current_aicc < best_aicc:
                best_aicc = current_aicc
                best_alpha = model_opt.params['smoothing_level']
                best_init = init
        except:
            continue
    return best_alpha, best_init

def create_features(series, lags=12):
    df = pd.DataFrame(series)
    col_name = df.columns[0]
    for l in range(1, lags + 1):
        df[f'lag_{l}'] = df[col_name].shift(l)
    df['month_num'] = df.index.month
    df = df.dropna()
    X = df.drop(columns=[col_name])
    y = df[col_name]
    return X, y

# ==========================================
# 2. ฟังก์ชันหลักของทั้ง 5 โมเดล
# ==========================================

# --- 1. SARIMA ---
def run_mdr_forecasting_sarima(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}\nAnalyzing: SARIMA | {target_drug_name}\n{'='*50}")
    train_size = int(len(series) * 0.80)
    train_data, test_data = series.iloc[:train_size], series.iloc[train_size:]

    print("Finding best SARIMA parameters (Full Grid Search)...")
    stepwise_model = auto_arima(
        train_data, start_p=0, start_q=0, max_p=5, max_q=5, start_P=0, start_Q=0, max_P=3, max_Q=3,
        m=12, seasonal=True, d=None, max_d=2, D=None, max_D=1, trace=False, error_action='ignore',
        suppress_warnings=True, stepwise=False, n_jobs=-1, information_criterion='aicc'
    )
    best_order, best_seasonal = stepwise_model.order, stepwise_model.seasonal_order
    
    model_eval = SARIMAX(train_data, order=best_order, seasonal_order=best_seasonal, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    rmse, wape = calculate_metrics(test_data, model_eval.forecast(steps=len(test_data)))
    
    final_model = SARIMAX(series, order=best_order, seasonal_order=best_seasonal, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    forecast_sarima = final_model.forecast(steps=forecast_months)

    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values, color='#377eb8', marker='o', markersize=4, label='Actual Data')
    conn_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    conn_val = np.concatenate([[series.values[-1]], forecast_sarima.values])
    plt.plot(conn_idx, conn_val, color='#e41a1c', marker='o', markersize=4, linestyle='--', label='SARIMA Forecast')
    plt.title(f'SARIMA Forecast (RMSE: {rmse:.2f}, WAPE: {wape:.2f}%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return rmse, wape

# --- 2. ARIMA ---
def run_mdr_forecasting_arima(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}\nAnalyzing: ARIMA | {target_drug_name}\n{'='*50}")
    train_size = int(len(series) * 0.80)
    train_data, test_data = series.iloc[:train_size], series.iloc[train_size:]

    print("Finding best ARIMA parameters...")
    stepwise_model = auto_arima(
        train_data, start_p=0, start_q=0, max_p=5, max_q=5, d=None, max_d=2, test='adf',
        seasonal=False, stepwise=False, information_criterion='aicc', n_jobs=-1, suppress_warnings=True, error_action='ignore', trace=False
    )
    best_order = stepwise_model.order
    
    model_eval = ARIMA(train_data, order=best_order).fit()
    rmse, wape = calculate_metrics(test_data, model_eval.forecast(steps=len(test_data)))
    
    final_model = ARIMA(series, order=best_order).fit()
    forecast_arima = final_model.forecast(steps=forecast_months)

    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values, color='#377eb8', marker='o', markersize=4, label='Actual Data')
    conn_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    conn_val = np.concatenate([[series.values[-1]], forecast_arima.values])
    plt.plot(conn_idx, conn_val, color='#e41a1c', marker='o', markersize=4, linestyle='--', label='ARIMA Forecast')
    plt.title(f'ARIMA Forecast (RMSE: {rmse:.2f}, WAPE: {wape:.2f}%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return rmse, wape

# --- 3. TES ---
def run_mdr_forecasting_tes(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}\nAnalyzing: TES (Holt-Winters) | {target_drug_name}\n{'='*50}")
    train_size = int(len(series) * 0.80)
    train_data, test_data = series.iloc[:train_size], series.iloc[train_size:]

    print("Finding best TES Configuration...")
    best_t, best_s, best_d = grid_search_tes(train_data)
    sp = 12 if best_s is not None else None
    
    model_eval = ExponentialSmoothing(train_data, trend=best_t, seasonal=best_s, seasonal_periods=sp, damped_trend=best_d, initialization_method="estimated").fit(optimized=True)
    rmse, wape = calculate_metrics(test_data, model_eval.forecast(len(test_data)))
    
    final_model = ExponentialSmoothing(series, trend=best_t, seasonal=best_s, seasonal_periods=sp, damped_trend=best_d, initialization_method="estimated").fit(optimized=True)
    forecast_tes = final_model.forecast(forecast_months)

    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values, color='#377eb8', marker='o', markersize=4, label='Actual Data')
    conn_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    conn_val = np.concatenate([[series.values[-1]], forecast_tes.values])
    plt.plot(conn_idx, conn_val, color='#e41a1c', marker='o', markersize=4, linestyle='--', label='TES Forecast')
    plt.title(f'TES Forecast (RMSE: {rmse:.2f}, WAPE: {wape:.2f}%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return rmse, wape

# --- 4. SES ---
def run_mdr_forecasting_ses(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}\nAnalyzing: SES | {target_drug_name}\n{'='*50}")
    train_size = int(len(series) * 0.80)
    train_data, test_data = series.iloc[:train_size], series.iloc[train_size:]

    print("Finding best SES Configuration...")
    best_alpha, best_init = grid_search_ses(train_data)
    
    model_eval = SimpleExpSmoothing(train_data, initialization_method=best_init).fit(smoothing_level=best_alpha, optimized=False)
    rmse, wape = calculate_metrics(test_data, model_eval.forecast(len(test_data)))
    
    final_model = SimpleExpSmoothing(series, initialization_method=best_init).fit(smoothing_level=best_alpha, optimized=False)
    forecast_ses = final_model.forecast(forecast_months)

    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values, color='#377eb8', marker='o', markersize=4, label='Actual Data')
    conn_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    conn_val = np.concatenate([[series.values[-1]], forecast_ses.values])
    plt.plot(conn_idx, conn_val, color='#e41a1c', marker='o', markersize=4, linestyle='--', label='SES Forecast')
    plt.title(f'SES Forecast (RMSE: {rmse:.2f}, WAPE: {wape:.2f}%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return rmse, wape

# --- 5. XGBoost ---
def run_mdr_forecasting_xgb(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}\nAnalyzing: XGBoost | {target_drug_name}\n{'='*50}")
    X, y = create_features(series, lags=12)
    train_size = int(len(X) * 0.80)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    print("Finding best XGBoost hyperparameters...")
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
    
    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
        param_distributions=param_distributions, n_iter=20, cv=TimeSeriesSplit(n_splits=3), # ลด n_iter เพื่อความรวดเร็วในการรันรวม
        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0, random_state=42
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    
    model_eval = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42).fit(X_train, y_train)
    rmse, wape = calculate_metrics(y_test, model_eval.predict(X_test))
    
    model_final = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42).fit(X, y)
    
    last_window = series.values[-12:].tolist() 
    future_forecast = []
    forecast_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

    for i in range(forecast_months):
        feature_values = last_window[::-1] + [forecast_dates[i].month]
        input_df = pd.DataFrame([feature_values], columns=X.columns)
        pred = max(0, model_final.predict(input_df)[0])
        future_forecast.append(pred)
        last_window.append(pred)
        last_window.pop(0)

    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values, color='#377eb8', marker='o', markersize=4, label='Actual Data')
    conn_idx = pd.to_datetime([series.index[-1]] + list(forecast_dates))
    conn_val = [series.values[-1]] + future_forecast
    plt.plot(conn_idx, conn_val, color='#e41a1c', marker='o', markersize=4, linestyle='--', label='XGBoost Forecast')
    plt.title(f'XGBoost Forecast (RMSE: {rmse:.2f}, WAPE: {wape:.2f}%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return rmse, wape

# ==========================================
# 3. ส่วนประมวลผลหลัก (Data Loading & Execution)
# ==========================================

if __name__ == "__main__":
    # ปรับ Path ชุดข้อมูลที่จะใช้เปรียบเทียบ (ใช้ชุดเดียวสำหรับทดสอบทุกโมเดล)
    file_path = os.path.join("MDR", "model","All Data", "acinetobacter_baumannii.csv") 
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
        all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
        full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
        
        final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
        final_df.index = all_months
        final_df = final_df.drop(columns=['year', 'month'])
        
        # จัดการ Missing Values ด้วยวิธีเดิมของคุณ 100%
        final_df = final_df.interpolate(method='linear')
        final_df = final_df.bfill().ffill()

        target_drug = 'CARBAPENEMS, CEPHEMS, FOLATE PATHWAY ANTAGONISTS, β-LACTAM COMBINATION AGENTS'

        if target_drug in final_df.columns:
            series_data = final_df[target_drug]
            display_name = "Pseudomonas aeruginosa (UR)"
            
            # Dictionary เก็บผลลัพธ์
            model_results = {}
            
            # รันและเก็บค่าจากทุกโมเดล
            rmse_sar, wape_sar = run_mdr_forecasting_sarima(series_data, display_name)
            model_results['SARIMA'] = {'RMSE': rmse_sar, 'WAPE (%)': wape_sar}
            
            rmse_ari, wape_ari = run_mdr_forecasting_arima(series_data, display_name)
            model_results['ARIMA'] = {'RMSE': rmse_ari, 'WAPE (%)': wape_ari}
            
            rmse_tes, wape_tes = run_mdr_forecasting_tes(series_data, display_name)
            model_results['TES (Holt-Winters)'] = {'RMSE': rmse_tes, 'WAPE (%)': wape_tes}
            
            rmse_ses, wape_ses = run_mdr_forecasting_ses(series_data, display_name)
            model_results['SES'] = {'RMSE': rmse_ses, 'WAPE (%)': wape_ses}
            
            rmse_xgb, wape_xgb = run_mdr_forecasting_xgb(series_data, display_name)
            model_results['XGBoost'] = {'RMSE': rmse_xgb, 'WAPE (%)': wape_xgb}
            
            # ==========================================
            # แสดงตารางสรุปผลเปรียบเทียบ
            # ==========================================
            print("\n" + "="*50)
            print("🚀 สรุปผลเปรียบเทียบประสิทธิภาพของทุกโมเดล (Model Comparison)")
            print("="*50)
            
            summary_df = pd.DataFrame(model_results).T
            
            # เรียงลำดับจาก WAPE ต่ำสุดไปหาสูงสุด (โมเดลที่ดีที่สุดอยู่บนสุด)
            summary_df = summary_df.sort_values(by='WAPE (%)')
            
            print(summary_df.to_string())
            print("="*50)
            print("💡 หมายเหตุ: โมเดลที่อยู่บนสุดคือโมเดลที่มีค่าความคลาดเคลื่อน (WAPE) ต่ำที่สุดสำหรับชุดข้อมูลนี้")
            
        else:
            print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
    else:
        print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")