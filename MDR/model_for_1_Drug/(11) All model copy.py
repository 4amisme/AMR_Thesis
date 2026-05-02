import pandas as pd
import numpy as np
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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

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
    trend_opts = ['add']
    seasonal_opts = ['add', None]
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
        except: continue
    return best_config if best_config else ('add', 'add', False)

def grid_search_ses(train_data):
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
# 2. ฟังก์ชันหลัก: รัน 5 โมเดล (แบบไม่โชว์กราฟ)
# ==========================================

def run_models_for_series(series):
    n_standard = len(series)
    train_size_std = int(n_standard * 0.80)
    train_data_std = series.iloc[:train_size_std]
    test_data_std = series.iloc[train_size_std:]
    
    model_results = []

    # 1. SARIMA
    try:
        sarima_auto = auto_arima(train_data_std, start_p=0, start_q=0, max_p=5, max_q=5, start_P=0, start_Q=0, max_P=3, max_Q=3, m=12, seasonal=True, d=None, max_d=2, D=None, max_D=1, trace=False, error_action='ignore', suppress_warnings=True, stepwise=False, n_jobs=-1, information_criterion='aicc')
        sarima_eval = SARIMAX(train_data_std, order=sarima_auto.order, seasonal_order=sarima_auto.seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred_sarima = sarima_eval.forecast(steps=len(test_data_std)).clip(0, 100)
        rmse, wape = calculate_metrics(test_data_std, pred_sarima)
        model_results.append({'Model': 'SARIMA', 'RMSE': rmse, 'WAPE': wape})
    except Exception as e:
        model_results.append({'Model': 'SARIMA', 'RMSE': np.nan, 'WAPE': np.nan})

    # 2. ARIMA
    try:
        arima_auto = auto_arima(train_data_std, start_p=0, start_q=0, max_p=5, max_q=5, seasonal=False, d=None, max_d=2, test='adf', stepwise=False, suppress_warnings=True, error_action='ignore', trace=False, n_jobs=-1, information_criterion='aicc')
        arima_eval = ARIMA(train_data_std, order=arima_auto.order).fit()
        pred_arima = arima_eval.forecast(steps=len(test_data_std)).clip(0, 100)
        rmse, wape = calculate_metrics(test_data_std, pred_arima)
        model_results.append({'Model': 'ARIMA', 'RMSE': rmse, 'WAPE': wape})
    except:
        model_results.append({'Model': 'ARIMA', 'RMSE': np.nan, 'WAPE': np.nan})

    # 3. TES
    try:
        best_t, best_s, best_d = grid_search_tes(train_data_std)
        sp = 12 if best_s is not None else None
        tes_eval = ExponentialSmoothing(train_data_std, trend=best_t, seasonal=best_s, seasonal_periods=sp, damped_trend=best_d, initialization_method="estimated").fit(optimized=True)
        pred_tes = tes_eval.forecast(len(test_data_std)).clip(0, 100)
        rmse, wape = calculate_metrics(test_data_std, pred_tes)
        model_results.append({'Model': 'TES', 'RMSE': rmse, 'WAPE': wape})
    except:
        model_results.append({'Model': 'TES', 'RMSE': np.nan, 'WAPE': np.nan})

    # 4. SES
    try:
        best_alpha, best_init = grid_search_ses(train_data_std)
        ses_eval = SimpleExpSmoothing(train_data_std, initialization_method=best_init).fit(smoothing_level=best_alpha, optimized=False)
        pred_ses = ses_eval.forecast(len(test_data_std)).clip(0, 100)
        rmse, wape = calculate_metrics(test_data_std, pred_ses)
        model_results.append({'Model': 'SES', 'RMSE': rmse, 'WAPE': wape})
    except:
        model_results.append({'Model': 'SES', 'RMSE': np.nan, 'WAPE': np.nan})

    # 5. XGBoost
    try:
        X, y = create_xgb_features(series, lags=12)
        train_size_xgb = int(len(X) * 0.80)
        X_train, y_train = X.iloc[:train_size_xgb], y.iloc[:train_size_xgb]
        X_test, y_test = X.iloc[train_size_xgb:], y.iloc[train_size_xgb:]

        param_distributions = {'n_estimators': [100, 300, 500], 'max_depth': [2, 3, 4], 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0], 'gamma': [0, 0.1], 'reg_alpha': [0, 1], 'reg_lambda': [1, 5]}
        xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42)
        random_search = RandomizedSearchCV(estimator=xgb_base, param_distributions=param_distributions, n_iter=15, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error', n_jobs=-1, verbose=0, random_state=42)
        random_search.fit(X_train, y_train)
        
        xgb_eval = XGBRegressor(**random_search.best_params_, objective='reg:squarederror', random_state=42).fit(X_train, y_train)
        pred_xgb = pd.Series(xgb_eval.predict(X_test), index=X_test.index).clip(0, 100)
        rmse, wape = calculate_metrics(y_test, pred_xgb)
        model_results.append({'Model': 'XGBoost', 'RMSE': rmse, 'WAPE': wape})
    except:
        model_results.append({'Model': 'XGBoost', 'RMSE': np.nan, 'WAPE': np.nan})

    return model_results

# ==========================================
# 3. ระบบวนลูปอ่านไฟล์และประมวลผล (Loop System)
# ==========================================

if __name__ == "__main__":
    
    # 📌 1. ระบุ Path ที่เกี่ยวข้อง
    reference_excel_path = r"C:\AMR_Thesis\MDR\model_for_1_Drug\1Drug.xlsx"
    
    target_folders = [
        r"C:\AMR_Thesis\MDR\model_for_1_Drug\All data",
        r"C:\AMR_Thesis\MDR\model_for_1_Drug\Specimen",
        r"C:\AMR_Thesis\MDR\model_for_1_Drug\Ward"
    ]
    
    # 📌 2. โหลดไฟล์ Excel อ้างอิง
    print(f"กำลังโหลดข้อมูลอ้างอิงจาก: {reference_excel_path}")
    try:
        ref_df = pd.read_excel(reference_excel_path)
    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์อ้างอิง: {reference_excel_path}")
        exit()

    all_results = []
    
    # 📌 3. วนลูปตามแต่ละโฟลเดอร์
    for folder in target_folders:
        if not os.path.exists(folder):
            print(f"⚠️ ไม่พบโฟลเดอร์: {folder} (ข้าม)")
            continue
            
        folder_name = os.path.basename(folder)
        print(f"\n📂 กำลังประมวลผลโฟลเดอร์: {folder_name}")
        
        # วนลูปอ่านทุกไฟล์ CSV ในโฟลเดอร์
        for file in os.listdir(folder):
            if not file.endswith('.csv'):
                continue
                
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path)
            
            # เช็คว่ามีคอลัมน์ organism_full หรือไม่
            if 'organism_full' not in df.columns:
                print(f"   ⚠️ ข้าม {file}: ไม่มีคอลัมน์ 'organism_full'")
                continue
                
            # หาชื่อเชื้อแบคทีเรียทั้งหมดที่มีในไฟล์นี้
            organisms_in_file = df['organism_full'].dropna().unique()
            
            # ดำเนินการ Pivot ข้อมูลเตรียมไว้ล่วงหน้า
            try:
                pivot_df = df.pivot_table(index=['year', 'month'], columns='resistant_drug_name', values='percentage', aggfunc='mean')
                all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
                full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
                final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
                final_df.index = all_months
                final_df = final_df.drop(columns=['year', 'month'])
                final_df = final_df.interpolate(method='linear').bfill().ffill()
            except Exception as e:
                print(f"   ❌ เกิดข้อผิดพลาดในการ Pivot ข้อมูลไฟล์ {file}: {e}")
                continue

            # วนลูปตามชื่อเชื้อที่พบในไฟล์ CSV
            for org in organisms_in_file:
                # 📌 เทียบชื่อเชื้อใน CSV กับคอลัมน์ Organism ใน Excel
                matched_rows = ref_df[ref_df['Organism'] == org]
                
                if matched_rows.empty:
                    # ถ้าไม่เจอชื่อเชื้อใน Excel ให้ข้าม
                    continue
                    
                # 📌 ดึงรายชื่อยาจากคอลัมน์ Drug ใน Excel ที่ตรงกับเชื้อนี้
                drugs_to_run = matched_rows['Drug'].dropna().unique()
                
                for target_drug in drugs_to_run:
                    if target_drug in final_df.columns:
                        print(f"   ⚙️ [รันโมเดล] ไฟล์: {file} | เชื้อ: {org} | ยา: {target_drug}")
                        series_data = final_df[target_drug]
                        
                        # โยนเข้าฟังก์ชันรัน 5 โมเดล
                        results = run_models_for_series(series_data)
                        
                        # เก็บผลลัพธ์
                        for res in results:
                            all_results.append({
                                'Source_Folder': folder_name,
                                'Source_File': file,
                                'Organism': org,
                                'Target_Drug': target_drug,
                                'Model': res['Model'],
                                'RMSE': res['RMSE'],
                                'WAPE': res['WAPE']
                            })
                    else:
                        print(f"   ⚠️ ไม่พบข้อมูลยา '{target_drug}' ในไฟล์ {file}")

    # ==========================================
    # 4. บันทึกผลลัพธ์และหา Best Model
    # ==========================================
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        
        # คัดกรองข้อมูลที่คำนวณ Error ไม่ได้ (NaN) ออกก่อนหา Best Model
        valid_results = results_df.dropna(subset=['WAPE'])
        
        if not valid_results.empty:
            # ค้นหาโมเดลที่มีค่า WAPE ต่ำที่สุดของแต่ละกลุ่ม
            best_models = valid_results.loc[valid_results.groupby(['Source_Folder', 'Source_File', 'Organism', 'Target_Drug'])['WAPE'].idxmin()][['Source_Folder', 'Source_File', 'Organism', 'Target_Drug', 'Model']]
            best_models.rename(columns={'Model': 'Best_Model_By_WAPE'}, inplace=True)
            
            # รวมตาราง
            final_summary_df = pd.merge(results_df, best_models, on=['Source_Folder', 'Source_File', 'Organism', 'Target_Drug'], how='left')
        else:
            final_summary_df = results_df

        # เซฟเป็นไฟล์ CSV ที่โฟลเดอร์หลัก
        output_csv_path = r"C:\AMR_Thesis\MDR\model_for_1_Drug\forecast_summary_1Drug_results.csv"
        final_summary_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print("🎉 ประมวลผลเสร็จสิ้นทุกไฟล์และทุกโฟลเดอร์!")
        print(f"💾 บันทึกผลลัพธ์สำเร็จที่: \n{output_csv_path}")
        print("="*60)
    else:
        print("\n⚠️ ไม่มีผลลัพธ์จากการประมวลผลเลย กรุณาตรวจสอบว่าชื่อ Organism ใน Excel ตรงกับใน CSV หรือไม่")