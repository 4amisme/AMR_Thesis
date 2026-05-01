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
# 2. ฟังก์ชันหลักของทั้ง 5 โมเดล (ปิดการพล็อตเพื่อให้ลูปรันต่อเนื่องได้)
# ==========================================

def run_mdr_forecasting_sarima(series, target_drug_name, forecast_months=60):
    train_size = int(len(series) * 0.80)
    train_data, test_data = series.iloc[:train_size], series.iloc[train_size:]

    stepwise_model = auto_arima(
        train_data, start_p=0, start_q=0, max_p=5, max_q=5, start_P=0, start_Q=0, max_P=3, max_Q=3,
        m=12, seasonal=True, d=None, max_d=2, D=None, max_D=1, trace=False, error_action='ignore',
        suppress_warnings=True, stepwise=False, n_jobs=-1, information_criterion='aicc'
    )
    best_order, best_seasonal = stepwise_model.order, stepwise_model.seasonal_order
    
    model_eval = SARIMAX(train_data, order=best_order, seasonal_order=best_seasonal, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    rmse, wape = calculate_metrics(test_data, model_eval.forecast(steps=len(test_data)))
    return rmse, wape

def run_mdr_forecasting_arima(series, target_drug_name, forecast_months=60):
    train_size = int(len(series) * 0.80)
    train_data, test_data = series.iloc[:train_size], series.iloc[train_size:]

    stepwise_model = auto_arima(
        train_data, start_p=0, start_q=0, max_p=5, max_q=5, d=None, max_d=2, test='adf',
        seasonal=False, stepwise=False, information_criterion='aicc', n_jobs=-1, suppress_warnings=True, error_action='ignore', trace=False
    )
    best_order = stepwise_model.order
    
    model_eval = ARIMA(train_data, order=best_order).fit()
    rmse, wape = calculate_metrics(test_data, model_eval.forecast(steps=len(test_data)))
    return rmse, wape

def run_mdr_forecasting_tes(series, target_drug_name, forecast_months=60):
    train_size = int(len(series) * 0.80)
    train_data, test_data = series.iloc[:train_size], series.iloc[train_size:]

    best_t, best_s, best_d = grid_search_tes(train_data)
    sp = 12 if best_s is not None else None
    
    model_eval = ExponentialSmoothing(train_data, trend=best_t, seasonal=best_s, seasonal_periods=sp, damped_trend=best_d, initialization_method="estimated").fit(optimized=True)
    rmse, wape = calculate_metrics(test_data, model_eval.forecast(len(test_data)))
    return rmse, wape

def run_mdr_forecasting_ses(series, target_drug_name, forecast_months=60):
    train_size = int(len(series) * 0.80)
    train_data, test_data = series.iloc[:train_size], series.iloc[train_size:]

    best_alpha, best_init = grid_search_ses(train_data)
    
    model_eval = SimpleExpSmoothing(train_data, initialization_method=best_init).fit(smoothing_level=best_alpha, optimized=False)
    rmse, wape = calculate_metrics(test_data, model_eval.forecast(len(test_data)))
    return rmse, wape

def run_mdr_forecasting_xgb(series, target_drug_name, forecast_months=60):
    X, y = create_features(series, lags=12)
    train_size = int(len(X) * 0.80)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    param_distributions = {
        'n_estimators': [100, 300, 500],
        'max_depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0.1, 1, 5]
    }
    
    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
        param_distributions=param_distributions, n_iter=10, cv=TimeSeriesSplit(n_splits=3), 
        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0, random_state=42
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    
    model_eval = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42).fit(X_train, y_train)
    rmse, wape = calculate_metrics(y_test, model_eval.predict(X_test))
    return rmse, wape

# ==========================================
# 3. ส่วนประมวลผลหลัก (Loop Processing)
# ==========================================

if __name__ == "__main__":
    
    # Path ที่ตั้งของไฟล์ Excel อ้างอิง
    reference_excel_path = "/Users/pammy/Downloads/AMR_Thesis/MDR/MDR.xlsx" 
    
    # Base Path ที่ตั้งของโฟลเดอร์หลักสำหรับโมเดล
    base_dir = "/Users/pammy/Downloads/AMR_Thesis/MDR/model"
    
    print(f"กำลังโหลดข้อมูลอ้างอิงจาก: {reference_excel_path}")
    try:
        ref_df = pd.read_excel(reference_excel_path)
    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์ Excel อ้างอิงที่: {reference_excel_path}")
        exit()

    # สร้าง List เพื่อเก็บผลลัพธ์ทั้งหมด
    all_results = []

    print(f"พบข้อมูลทั้งหมด {len(ref_df)} รายการ เริ่มกระบวนการประมวลผล...\n" + "="*50)

    # วนลูปตามแต่ละบรรทัดในไฟล์อ้างอิง Excel
    for index, row in ref_df.iterrows():
        # ตรวจสอบค่าว่าง (NaN) และแปลงเป็น string
        if pd.isna(row.get('Source_Folder')) or pd.isna(row.get('Source_File')):
            continue
            
        source_folder = str(row['Source_Folder']).strip()
        source_file = str(row['Source_File']).strip()
        target_drug = str(row['Resistant_Drug_Classes']).strip()
        
        # สมมติชื่อแบคทีเรียจากชื่อไฟล์ (ถ้าใน Excel มีคอลัมน์ชื่อนี้ ให้เปลี่ยนไปดึงจาก row ได้เลย)
        bacteria_name = source_file.replace('.csv', '').replace('_', ' ').title()

        # สร้าง Path สำหรับไปอ่านไฟล์ โดยเอา base_dir + Source_Folder + Source_File
        file_path = os.path.join(base_dir, source_folder, source_file)
        
        print(f"[{index+1}/{len(ref_df)}] 📂 โฟลเดอร์: {source_folder} | 📄 ไฟล์: {source_file}")
        print(f"          💊 ยา: {target_drug[:40]}...")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # แปลงโครงสร้างข้อมูลและสร้าง Time Series
            try:
                pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
                all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
                full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
                
                final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
                final_df.index = all_months
                final_df = final_df.drop(columns=['year', 'month'])
                
                # จัดการ Missing Values (Interpolate 100%)
                final_df = final_df.interpolate(method='linear').bfill().ffill()
            except Exception as e:
                print(f"   ❌ ข้าม: เกิดข้อผิดพลาดในการเตรียมข้อมูล ({e})")
                continue

            if target_drug in final_df.columns:
                series_data = final_df[target_drug]
                
                # รันแต่ละโมเดลและเก็บค่า
                models_to_run = {
                    'SARIMA': run_mdr_forecasting_sarima,
                    'ARIMA': run_mdr_forecasting_arima,
                    'TES': run_mdr_forecasting_tes,
                    'SES': run_mdr_forecasting_ses,
                    'XGBoost': run_mdr_forecasting_xgb
                }
                
                for model_name, model_func in models_to_run.items():
                    try:
                        rmse, wape = model_func(series_data, bacteria_name)
                        
                        # เพิ่มข้อมูลลงใน List สรุป
                        all_results.append({
                            'Source_Folder': source_folder,
                            'Source_File': source_file,
                            'Bacteria_Name': bacteria_name,
                            'Target_Drug': target_drug,
                            'Model': model_name,
                            'RMSE': rmse,
                            'WAPE': wape
                        })
                        print(f"   ✅ {model_name:<10} -> RMSE: {rmse:>7.4f}, WAPE: {wape:>7.4f}%")
                    except Exception as e:
                        print(f"   ❌ ข้าม: {model_name} ประมวลผลไม่สำเร็จ ({e})")
            else:
                print(f"   ⚠️ ไม่พบกลุ่มยา '{target_drug}' ในไฟล์ข้อมูล")
        else:
            print(f"   ⚠️ ไม่พบไฟล์ CSV ตามที่ระบุ: {file_path}")
            
    # ==========================================
    # 4. บันทึกผลลัพธ์ทั้งหมดลงไฟล์ CSV
    # ==========================================
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        
        # ค้นหาโมเดลที่ดีที่สุด (Best Model based on WAPE) ของแต่ละไฟล์/กลุ่มยา
        best_models = results_df.loc[results_df.groupby(['Source_Folder', 'Source_File', 'Target_Drug'])['WAPE'].idxmin()][['Source_File', 'Target_Drug', 'Model']]
        best_models.rename(columns={'Model': 'Best_Model_By_WAPE'}, inplace=True)
        
        # นำคอลัมน์ Best Model มาต่อเชื่อมในตารางสรุป
        final_summary_df = pd.merge(results_df, best_models, on=['Source_File', 'Target_Drug'], how='left')

        # บันทึกไฟล์ผลลัพธ์
        output_csv_path = os.path.join(base_dir, "forecast_summary_results.csv")
        final_summary_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(f"🎉 ประมวลผลเสร็จสิ้นทั้งหมด!")
        print(f"💾 บันทึกผลลัพธ์รวบยอดไว้ที่: \n{output_csv_path}")
        print("="*50)
    else:
        print("\n⚠️ ไม่มีผลลัพธ์จากการประมวลผลเลย (กรุณาตรวจสอบชื่อไฟล์ หรือชื่อโฟลเดอร์ใน Excel)")