import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import warnings

# --- [NEW] เพิ่ม Import สำหรับทำ Residual Plot ---
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf

# ปิดการแจ้งเตือนเพื่อให้ Output สะอาด
warnings.filterwarnings("ignore")

# ==========================================
# 1. ฟังก์ชันคำนวณ Metrics และ Tuning
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE สำหรับวัดประสิทธิภาพ"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sum_true = np.sum(y_true)
    wape = np.sum(np.abs(y_true - y_pred)) / sum_true * 100 if sum_true != 0 else 0
    return round(rmse, 4), round(wape, 4)

def grid_search_ses(train_data):
    """ทำ Parameter Tuning เพื่อหาค่า Alpha และ Initialization ที่ดีที่สุดด้วย AICc"""
    best_aicc = float('inf')
    best_alpha = None
    best_init = None
    
    # ทดสอบค่า Alpha ตั้งแต่โมเดลที่ตอบสนองช้า (0.01) ไปจนถึงไวมาก (0.99)
    alphas = np.arange(0.01, 1.0, 0.05)
    init_methods = ['estimated', 'heuristic']
    
    for init in init_methods:
        # 1. ลองบังคับค่า Alpha ทีละสเตป
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
                
        # 2. ลองให้ระบบ Optimize หาค่าที่ดีที่สุดในวิถีของมันด้วย
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

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์และทำนาย (SES)
# ==========================================

def run_mdr_forecasting_ses(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}")
    print(f"Analyzing (Simple Exponential Smoothing): {target_drug_name}")
    print(f"{'='*50}")

    # --- [A] การแบ่งข้อมูล (80/20 สำหรับ Validation) ---
    n = len(series)
    train_size = int(n * 0.80)
    train_data = series.iloc[:train_size]
    test_data = series.iloc[train_size:]

    # --- [B] Parameter Tuning & Model Training ---
    print("\n[กำลังค้นหาค่า Alpha และวิธี Initialization ที่ดีที่สุดด้วย AICc...]")
    best_alpha, best_init = grid_search_ses(train_data)
    print(f">>> Best Alpha: {best_alpha:.4f}")
    print(f">>> Best Init Method: {best_init}")

    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
    model_eval = SimpleExpSmoothing(train_data, initialization_method=best_init).fit(smoothing_level=best_alpha, optimized=False)
    
    test_pred_ses = model_eval.forecast(len(test_data))
    rmse, wape = calculate_metrics(test_data, test_pred_ses)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (100% Data) ---")
    # นำ Alpha และ Init ที่ดีที่สุดมาเทรนกับข้อมูล 100%
    final_model = SimpleExpSmoothing(series, initialization_method=best_init).fit(smoothing_level=best_alpha, optimized=False)
    
    # ทำนายล่วงหน้า 5 ปี (หมายเหตุ: SES จะพยากรณ์เป็นเส้นตรงเสมอเนื่องจากไม่มี Trend)
    forecast_ses = final_model.forecast(forecast_months)

    # --- [NEW] Plotting Residual Diagnostics (Manual for SES) ---
    print("\n--- 3. Plotting Residual Diagnostics ---")
    residuals = final_model.resid
    
    fig_diag, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig_diag.suptitle(f'Residual Diagnostics (SES): {target_drug_name}', fontsize=14, y=1.02)
    
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

    # --- [C] การพล็อตแสดงผล ---
    
    plt.figure(figsize=(12, 6))
    
    # ข้อมูลจริง
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data (Interpolated)', linewidth=1.5)
    
    # เส้นพยากรณ์อนาคต
    conn_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    conn_val = np.concatenate([[series.values[-1]], forecast_ses.values])
    
    plt.plot(conn_idx, conn_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'Forecast (Next 5 years)', linewidth=1.5)

    plt.title(f'{target_drug_name} Multidrug-Resistant Forecast', 
              fontsize=14, fontweight='bold', pad=30) 
    plt.text(0.5, 1.03, f'Model: SES | Evaluation: (RMSE: {rmse:.2f}, WAPE: {wape:.2f}%)', 
             fontsize=11, ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.xlabel('Year')
    plt.ylabel('Resistance Percentage (%R)')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. ส่วนการรันข้อมูล
# ==========================================

file_path = os.path.join("MDR", "model","By_specimen", "p_aeruginosa_sp.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    final_df.index = all_months

    # ลบคอลัมน์ที่ไม่ใช่ข้อมูลเป้าหมายออกก่อนทำการ interpolate
    final_df = final_df.drop(columns=['year', 'month'])
    
    # ใช้ Linear Interpolation เพื่อประมาณค่าเดือนที่หายไปตามแนวโน้ม
    final_df = final_df.interpolate(method='linear')
    
    # ใช้ bfill และ ffill เพื่อจัดการกรณีค่าว่างที่หัวและท้ายตารางที่ interpolate เข้าไม่ถึง
    final_df = final_df.bfill().ffill()

    target_drug = 'AMINOGLYCOSIDES, CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        # [แก้ชื่อให้ตรงกับไฟล์]: เปลี่ยนเป็น Klebsiella pneumoniae
        run_mdr_forecasting_ses(series_data, "Pseudomonas aeruginosa")
    else:
        print(f"ไม่พบกลุ่มยา: {target_drug}")
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")