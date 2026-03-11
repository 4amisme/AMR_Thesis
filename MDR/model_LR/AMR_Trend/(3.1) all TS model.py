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
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend'

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
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
    
    best_name = min(all_metrics, key=lambda k: all_metrics[k]['RMSE'])
    return best_name, all_metrics[best_name], best_params.get(best_name, {}), all_metrics

def plot_residual_analysis(series, best_model, params, title, save_dir):
    """
    ฟังก์ชันวิเคราะห์ Residual Analysis แบบ Robust 
    เพื่อตรวจสอบความแม่นยำของโมเดลโดยไม่พึ่งพาคำสั่งที่มีปัญหาใน Seaborn
    """
    import numpy as np
    import scipy.stats as stats
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import os
    from xgboost import XGBRegressor # ตรวจสอบให้แน่ใจว่าได้ import ไว้ด้านบนของไฟล์แล้ว

    try:
        # 1. คำนวณ Residuals ตามประเภทของ Best Model
        if best_model == 'ARIMA':
            model_fit = sm.tsa.ARIMA(series, order=params['order']).fit()
            residuals = model_fit.resid
        elif best_model == 'SARIMA':
            model_fit = sm.tsa.SARIMAX(series, order=params['order'], 
                                       seasonal_order=params['seasonal_order']).fit(disp=False)
            residuals = model_fit.resid
        elif best_model in ['SES', 'TES']:
            if best_model == 'SES':
                model_fit = sm.tsa.SimpleExpSmoothing(series, initialization_method="estimated").fit(optimized=True)
            else:
                model_fit = sm.tsa.ExponentialSmoothing(series, trend='add', seasonal='add', 
                                                       seasonal_periods=12).fit(optimized=True)
            residuals = series - model_fit.fittedvalues
        elif best_model == 'XGBoost':
            X, y = create_xgb_features(series)
            xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X, y)
            residuals = y - xgb.predict(X)
        else:
            return
    except Exception as e:
        print(f"⚠️ ไม่สามารถวิเคราะห์ Residual สำหรับ {title}: {e}")
        return

    # 2. เริ่มขั้นตอนการ Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Residual Analysis: {title} (Model: {best_model})', fontsize=16)

    # ข้อมูลสำหรับการ Plot (ลบค่าว่างออก)
    res_data = residuals.dropna()
    
    # 🌟 จุดที่แก้ไข: แปลง Index และ Values ให้เป็น Numpy Array ก่อนนำไปพล็อต
    x_vals = res_data.index.to_numpy()
    y_vals = res_data.to_numpy()

    # --- Plot 1: Residuals over Time ---
    axes[0, 0].plot(x_vals, y_vals, color='gray', alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_title('Residuals over Time')
    axes[0, 0].tick_params(axis='x', rotation=45)

    if len(y_vals) > 0:
        # วาด Histogram (ใช้ y_vals ที่เป็น Numpy แล้วเพื่อความปลอดภัย)
        axes[0, 1].hist(y_vals, bins=20, density=True, color='skyblue', edgecolor='black', alpha=0.6)
        # คำนวณและวาดเส้น KDE
        try:
            kde = stats.gaussian_kde(y_vals)
            x_range = np.linspace(y_vals.min(), y_vals.max(), 100)
            axes[0, 1].plot(x_range, kde(x_range), color='darkblue', linewidth=2, label='KDE')
        except:
            pass
    axes[0, 1].set_title('Residual Distribution (Histogram & KDE)')

    # --- Plot 3: Q-Q Plot (ตรวจสอบการกระจายตัวแบบปกติ) ---
    stats.probplot(y_vals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q Plot')

    # --- Plot 4: ACF Plot (ตรวจสอบความสัมพันธ์ในตัวเองที่หลงเหลือ) ---
    sm.graphics.tsa.plot_acf(y_vals, lags=min(24, len(y_vals)//2), ax=axes[1, 1])
    axes[1, 1].set_title('Residual Autocorrelation (ACF)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 3. บันทึกรูปภาพ
    save_path = os.path.join(save_dir, f'Residual_{title.replace("/", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ บันทึก Residual Plot สำเร็จ: {save_path}")

def process_forecasting(df, category_folder, group_col=None):
    df = df.replace([np.inf, -np.inf], np.nan)
    df['date'] = pd.to_datetime(df['date'])
    full_idx = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    groups = [None] if group_col is None else df[group_col].dropna().unique()
    
    forecast_summaries = []
    
    for g in groups:
        for drug in ['vancomycin']:
            name = drug if g is None else f"{g}_{drug}"
            title = f"{category_folder} - {name}"
            print(f"​Modeling: {title} ...")
            
            temp = df[df['drug'] == drug.capitalize()]
            if group_col: temp = temp[temp[group_col] == g]
            if temp.empty: continue
            
            series = temp.set_index('date')['percent_R'].reindex(full_idx).interpolate(method='linear').fillna(0)
            
            train_size = int(len(series) * 0.8)
            train, test = series.iloc[:train_size], series.iloc[train_size:]
            
            best_model, best_metrics, params, all_metrics = find_best_model(train, test, series)
            if best_model is None: continue

            # Residual Plots โดยเฉพาะ
            residual_dir = os.path.join(BASE_DIR, category_folder, 'Residual_Analysis')
            os.makedirs(residual_dir, exist_ok=True)
            plot_residual_analysis(series, best_model, params, name, residual_dir)
            
            # 5 years
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

            plt.figure(figsize=(10, 5))
            plt.plot(series.index.to_numpy(), series.values, color='#1f77b4', marker='o', markersize=3, label='Actual (2015-2024)', linewidth=1.5)
            
            forecast_idx = pd.date_range(start=series.index[-1], periods=61, freq='MS')
            forecast_val = np.concatenate([[series.values[-1]], pred])
        
            plt.plot(forecast_idx.to_numpy(), forecast_val, color='#d62728', marker='o', markersize=3, linestyle='--', 
                     label=f'Forecast ({best_model})', linewidth=1.5)
            
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
    all_results = []
    
    file_all = os.path.join(BASE_DIR, 'All data', 'Data', 'efa_monthly_overall.csv')
    if os.path.exists(file_all):
        all_results.extend(process_forecasting(pd.read_csv(file_all), 'All data', None))
        
    file_ward = os.path.join(BASE_DIR, 'by ward', 'Data', 'monthly_ward.csv')
    if os.path.exists(file_ward):
        all_results.extend(process_forecasting(pd.read_csv(file_ward), 'by ward', 'ward_type'))
        
    file_spec = os.path.join(BASE_DIR, 'by specimen', 'Data', 'efa_monthly_specimen.csv')
    if os.path.exists(file_spec):
        df_spec = pd.read_csv(file_spec)
        df_spec = df_spec[df_spec['spec_group'] != 'other']
        all_results.extend(process_forecasting(df_spec, 'by specimen', 'spec_group'))

    if all_results:
        summary_df = pd.DataFrame(all_results)
        save_csv = os.path.join(BASE_DIR, 'efa_Forecasting_Models_Comprehensive_Evaluation.csv')
        summary_df.to_csv(save_csv, index=False)
        print(f" \n {save_csv}")

if __name__ == "__main__":
    step4_run_forecasting()