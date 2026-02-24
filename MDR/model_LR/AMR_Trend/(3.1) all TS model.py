import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend'

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # คำนวณ RMSE และ MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # คำนวณ WAPE
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    
    # คำนวณ MAPE (ป้องกัน Error กรณีหารด้วย 0)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0
    
    return {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'WAPE': round(wape, 4),
        'MAPE': round(mape, 4)
    }

def create_xgb_features(series):
    df = pd.DataFrame(series)
    col = df.columns[0]
    for l in range(1, 13):
        df[f'lag_{l}'] = df[col].shift(l)
    df['month'] = df.index.month
    df = df.dropna()
    return df.drop(columns=[col]), df[col]

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์ ทำจูน และทำนาย
# ==========================================
def find_best_model(train, test, series):
    all_metrics = {}
    best_params = {}
    
    # 1. ARIMA
    try:
        mod_arima = auto_arima(train, seasonal=False, suppress_warnings=True, error_action='ignore')
        pred = ARIMA(train, order=mod_arima.order).fit().forecast(len(test))
        all_metrics['ARIMA'] = calculate_metrics(test, pred)
        best_params['ARIMA'] = {'order': mod_arima.order}
    except: pass
    
    # 2. SARIMA
    try:
        mod_sarima = auto_arima(train, seasonal=True, m=12, suppress_warnings=True, error_action='ignore')
        pred = SARIMAX(train, order=mod_sarima.order, seasonal_order=mod_sarima.seasonal_order).fit(disp=False).forecast(len(test))
        all_metrics['SARIMA'] = calculate_metrics(test, pred)
        best_params['SARIMA'] = {'order': mod_sarima.order, 'seasonal_order': mod_sarima.seasonal_order}
    except: pass
    
    # 3. SES
    try:
        pred = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True).forecast(len(test))
        all_metrics['SES'] = calculate_metrics(test, pred)
    except: pass
    
    # 4. TES
    try:
        pred = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit(optimized=True).forecast(len(test))
        all_metrics['TES'] = calculate_metrics(test, pred)
    except: pass
    
    # 5. XGBoost
    try:
        X, y = create_xgb_features(series)
        split_idx = int(len(X) * 0.8)
        X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
        X_te, y_te = X.iloc[split_idx:], y.iloc[split_idx:]
        
        xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_tr, y_tr)
        all_metrics['XGBoost'] = calculate_metrics(y_te, xgb.predict(X_te))
    except: pass

    if not all_metrics: return None, None, None, None
    
    # 🌟 เปลี่ยนเกณฑ์การตัดสินให้ใช้ RMSE (ค่าน้อยที่สุด) เป็นตัวชี้วัดหลัก
    best_name = min(all_metrics, key=lambda k: all_metrics[k]['RMSE'])
    return best_name, all_metrics[best_name], best_params.get(best_name, {}), all_metrics

def process_forecasting(df, category_folder, group_col=None):
    df['date'] = pd.to_datetime(df['date'])
    full_idx = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    groups = [None] if group_col is None else df[group_col].dropna().unique()
    
    forecast_summaries = []
    
    for g in groups:
        for drug in ['Imipenem', 'Meropenem']:
            name = drug if g is None else f"{g}_{drug}"
            title = f"{category_folder} - {name}"
            print(f"กำลังหาโมเดล: {title} ...")
            
            temp = df[df['drug'] == drug.capitalize()]
            if group_col: temp = temp[temp[group_col] == g]
            if temp.empty: continue
            
            series = temp.set_index('date')['percent_R'].reindex(full_idx).interpolate(method='linear').fillna(0)
            
            train_size = int(len(series) * 0.8)
            train, test = series.iloc[:train_size], series.iloc[train_size:]
            
            best_model, best_metrics, params, all_metrics = find_best_model(train, test, series)
            if best_model is None: continue
            
            # พยากรณ์ 60 เดือน
            pred = []
            if best_model == 'ARIMA':
                pred = ARIMA(series, order=params['order']).fit().forecast(steps=60).values
            elif best_model == 'SARIMA':
                pred = SARIMAX(series, order=params['order'], seasonal_order=params['seasonal_order']).fit(disp=False).forecast(steps=60).values
            elif best_model == 'SES':
                pred = SimpleExpSmoothing(series, initialization_method="estimated").fit(optimized=True).forecast(60).values
            elif best_model == 'TES':
                pred = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12).fit(optimized=True).forecast(60).values
            elif best_model == 'XGBoost':
                X, y = create_xgb_features(series)
                xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X, y)
                last_w = series.values[-12:].tolist()
                dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=60, freq='MS')
                for i in range(60):
                    p = max(0, xgb.predict(pd.DataFrame([last_w[::-1] + [dates[i].month]], columns=X.columns))[0])
                    pred.append(p); last_w.append(p); last_w.pop(0)

            # วาดกราฟ
            plt.figure(figsize=(10, 5))
            plt.plot(series.index.to_numpy(), series.values, color='#1f77b4', marker='o', markersize=3, label='Actual (2015-2024)', linewidth=1.5)
            
            forecast_idx = pd.date_range(start=series.index[-1], periods=61, freq='MS')
            forecast_val = np.concatenate([[series.values[-1]], pred])
        
            plt.plot(forecast_idx.to_numpy(), forecast_val, color='#d62728', marker='o', markersize=3, linestyle='--', 
                     label=f'Forecast ({best_model})', linewidth=1.5)
            
            # 🌟 อัปเดต Title ให้แสดง RMSE (เป็นตัวตัดสิน) คู่กับ WAPE
            best_rmse, best_wape = best_metrics['RMSE'], best_metrics['WAPE']
            plt.title(f'AMR %R Forecast: {title}\nBest Model: {best_model} (RMSE = {best_rmse:.2f} | WAPE = {best_wape:.2f}%)', fontsize=12)
            
            plt.xlabel('Year')
            plt.ylabel('Resistance Percentage (%R)')
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='upper left')
            plt.tight_layout()
            
            save_path = os.path.join(BASE_DIR, category_folder, 'Forecasts', f'Forecast_{name.replace("/", "_")}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            # 🌟 สร้าง Row ข้อมูลสรุปเก็บเข้า List (ยึดตาม RMSE)
            row_data = {
                'Category': category_folder,
                'Group': g if g else 'Overall',
                'Drug': drug,
                'Best_Model': best_model,
                'RMSE': best_metrics['RMSE'],
                'WAPE': best_metrics['WAPE'],
                'MAE': best_metrics['MAE'],
                'MAPE': best_metrics['MAPE']
            }
            
            # เติมค่า 4 Metrics สำหรับทุกๆ โมเดล
            models_list = ['ARIMA', 'SARIMA', 'TES', 'SES', 'XGBoost']
            for m in models_list:
                m_data = all_metrics.get(m, {'WAPE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan})
                row_data[f'{m}_RMSE'] = m_data['RMSE']
                row_data[f'{m}_WAPE'] = m_data['WAPE']
                row_data[f'{m}_MAE'] = m_data['MAE']
                row_data[f'{m}_MAPE'] = m_data['MAPE']
                
            forecast_summaries.append(row_data)
            
    return forecast_summaries

def step4_run_forecasting():
    print("⏳ Step 4: Running Auto-ML Forecasting (Ranked by RMSE)...")
    all_results = []
    
    file_all = os.path.join(BASE_DIR, 'All data', 'Data', 'monthly_overall.csv')
    if os.path.exists(file_all):
        all_results.extend(process_forecasting(pd.read_csv(file_all), 'All data', None))
        
    file_ward = os.path.join(BASE_DIR, 'by ward', 'Data', 'monthly_ward.csv')
    if os.path.exists(file_ward):
        all_results.extend(process_forecasting(pd.read_csv(file_ward), 'by ward', 'ward_type'))
        
    file_spec = os.path.join(BASE_DIR, 'by specimen', 'Data', 'monthly_specimen.csv')
    if os.path.exists(file_spec):
        df_spec = pd.read_csv(file_spec)
        df_spec = df_spec[df_spec['spec_group'] != 'other']
        all_results.extend(process_forecasting(df_spec, 'by specimen', 'spec_group'))

    if all_results:
        summary_df = pd.DataFrame(all_results)
        save_csv = os.path.join(BASE_DIR, 'Forecasting_Models_Comprehensive_Evaluation.csv')
        summary_df.to_csv(save_csv, index=False)
        print(f"\n✅ Step 4 Complete! ตารางเปรียบเทียบเชิงลึกถูกบันทึกไว้ที่:\n   {save_csv}")

if __name__ == "__main__":
    step4_run_forecasting()