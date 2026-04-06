import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ปิดแจ้งเตือน Warning เพื่อให้หน้าจอ Output อ่านง่ายขึ้น
warnings.simplefilter("ignore")

# ==========================================
# 1. ฟังก์ชันคำนวณ Metrics
# ==========================================

def calculate_metrics(y_true, y_pred):
    """คำนวณ RMSE และ WAPE สำหรับวัดประสิทธิภาพ"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # ป้องกันการหารด้วยศูนย์
    sum_true = np.sum(y_true)
    wape = (np.sum(np.abs(y_true - y_pred)) / sum_true * 100) if sum_true != 0 else 0
    return round(rmse, 4), round(wape, 4)

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์และทำนาย
# ==========================================

def run_mdr_forecasting(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}")
<<<<<<< HEAD
    print(f"Analyzing: {target_drug_name}")
=======
    print(f"Analyzing (SARIMA): {target_drug_name}")
>>>>>>> Pammy
    print(f"{'='*50}")

    # --- [A] การแบ่งข้อมูล (80/20 สำหรับ Validation) ---
    n = len(series)
    train_size = int(n * 0.80)
    train_data = series.iloc[:train_size]
    test_data = series.iloc[train_size:]

    # --- [B] วิเคราะห์ ACF / PACF ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(series, lags=24, ax=ax1)
    ax1.set_title(f'ACF: {target_drug_name}')
    plot_pacf(series, lags=24, ax=ax2)
    ax2.set_title(f'PACF: {target_drug_name}')
    plt.show()

<<<<<<< HEAD
    # --- [C] Parameter Tuning ด้วย auto_arima (อัปเดตใหม่) ---
=======
    # --- [C] Parameter Tuning ด้วย auto_arima ---
>>>>>>> Pammy
    print("Finding best SARIMA parameters (Full Grid Search)...")
    
    # ปรับจูน auto_arima เพื่อลด RMSE/WAPE
    stepwise_model = auto_arima(
        train_data, 
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        start_P=0, start_Q=0,
        max_P=3, max_Q=3,          # เพิ่มขอบเขตการค้นหาฝั่ง Seasonality
        m=12,                      # ข้อมูลรายเดือน (Seasonality = 12)
        seasonal=True,
        d=None,                    # ให้ระบบหาค่า d เอง (0 ถึง 2)
        max_d=2,
        D=None,                    # ให้ระบบหาค่า D เอง (สำคัญมาก ช่วยลด Over-differencing)
        max_D=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
<<<<<<< HEAD
        stepwise=False,            # [สำคัญ] ปิด Stepwise เพื่อบังคับให้หาทุกรูปแบบ (ใช้เวลาเพิ่มขึ้นแต่มักได้ผลลัพธ์ดีกว่า)
        n_jobs=-1,                 # ใช้ทุก Core ของ CPU เพื่อให้ค้นหาเร็วขึ้น
        information_criterion='aicc' # ใช้ AICc ซึ่งเหมาะกับข้อมูล Time Series ขนาดเล็ก-กลาง มากกว่า AIC ปกติ
=======
        stepwise=False,            # [สำคัญ] ปิด Stepwise เพื่อบังคับให้หาทุกรูปแบบ
        n_jobs=-1,                 # ใช้ทุก Core ของ CPU เพื่อให้ค้นหาเร็วขึ้น
        information_criterion='aicc' # ใช้ AICc ซึ่งเหมาะกับข้อมูล Time Series ขนาดเล็ก-กลาง
>>>>>>> Pammy
    )

    best_order = stepwise_model.order
    best_seasonal = stepwise_model.seasonal_order
    
    print(f"\n>>> Best Order: {best_order}")
    print(f">>> Best Seasonal Order: {best_seasonal}")
    print(f">>> Best AICc: {stepwise_model.aic():.2f}")

    # --- [D] Model Training & Forecasting ---
    
    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
    model_eval = SARIMAX(train_data, 
                         order=best_order, 
                         seasonal_order=best_seasonal,
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
    
<<<<<<< HEAD
    test_pred_sarima = model_eval.forecast(steps=len(test_data))
=======
    # [UPDATE] ล็อกค่าพยากรณ์ช่วง Test ให้อยู่ระหว่าง 0-100%
    test_pred_sarima = model_eval.forecast(steps=len(test_data)).clip(0, 100)
>>>>>>> Pammy
    rmse, wape = calculate_metrics(test_data, test_pred_sarima)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (Full Data) ---")
    final_model = SARIMAX(series, 
                          order=best_order, 
                          seasonal_order=best_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
    
<<<<<<< HEAD
    forecast_sarima = final_model.forecast(steps=forecast_months)
=======
    # [UPDATE] ล็อกค่าพยากรณ์อนาคตให้อยู่ระหว่าง 0-100%
    forecast_sarima = final_model.forecast(steps=forecast_months).clip(0, 100)
>>>>>>> Pammy

    # --- [NEW] Plotting Residual Diagnostics ---
    print("\n--- 3. Plotting Residual Diagnostics ---")
    fig_diag = final_model.plot_diagnostics(figsize=(15, 8))
<<<<<<< HEAD
    fig_diag.suptitle(f'Residual Diagnostics: {target_drug_name}', fontsize=14, y=1.02)
=======
    fig_diag.suptitle(f'Residual Diagnostics (SARIMA): {target_drug_name}', fontsize=14, y=1.02)
>>>>>>> Pammy
    plt.tight_layout()
    plt.show()

    # --- [E] การพล็อตแสดงผล ---
    
    plt.figure(figsize=(12, 6))
    
    # พล็อตข้อมูลจริง
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data (Interpolated)', linewidth=1.5)
    
<<<<<<< HEAD
    # พล็อตเส้นเชื่อมและ Forecast (สีแดง)
=======
    # พล็อตเส้นเชื่อมและ Forecast
>>>>>>> Pammy
    forecast_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    forecast_val = np.concatenate([[series.values[-1]], forecast_sarima.values])
    
    plt.plot(forecast_idx, forecast_val, 
<<<<<<< HEAD
             color='#e41a1c', marker='o', markersize=4, linestyle='--', label='Forecast (Next 5 years)', linewidth=1.5)

    # ตกแต่งกราฟ
    plt.title(f'{target_drug_name} Multidrug-Resistant Forecast', 
              fontsize=14, fontweight='bold', pad=30) 
=======
             color='#e41a1c', marker='o', markersize=4, linestyle='--', 
             label=f'Forecast (Next {forecast_months//12} years)', linewidth=1.5)

    # ตกแต่งกราฟ
    plt.title(f'{target_drug_name} Forecast', fontsize=14, fontweight='bold', pad=30) 
>>>>>>> Pammy
    plt.text(0.5, 1.03, f'Model: SARIMA | Evaluation: (RMSE: {rmse:.2f}, WAPE: {wape:.2f}%)', 
             fontsize=11, ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.xlabel('Year')
    plt.ylabel('Resistance Percentage (%R)')
<<<<<<< HEAD
=======
    plt.ylim(0, max(100, series.max() + 5)) # [UPDATE] ล็อกสเกลแกน Y ให้ครอบคลุม 0-100 อย่างสวยงาม
>>>>>>> Pammy
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. ส่วนการรันข้อมูล (Data Preprocessing)
# ==========================================

<<<<<<< HEAD
# ปรับ Path ตามที่อยู่ไฟล์จริง
file_path = os.path.join("MDR", "model_1Class","All Data", "pseudomonas_aeruginosa_.csv") 
=======
# [UPDATE] ปรับ Path ให้ตรงกับไฟล์ S. aureus ของคุณ
file_path = os.path.join("MDR", "model_for_1_Drug", "Specimen", "e_coli_sp.csv") 
>>>>>>> Pammy

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
<<<<<<< HEAD
    # 1. เตรียมข้อมูล Wide Format
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
=======
    # [UPDATE] ใช้ resistant_drug_name และคำนวณค่าเฉลี่ยหากมีหลายเคสในเดือนเดียวกัน
    pivot_df = df.pivot_table(
        index=['year', 'month'], 
        columns='resistant_drug_name', 
        values='percentage',
        aggfunc='mean'
    )
>>>>>>> Pammy
    
    # สร้าง Index วันที่ให้สมบูรณ์ (2015-2024)
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
<<<<<<< HEAD
    # Merge ข้อมูลเข้ากับตารางวันที่หลัก (เดือนไหนไม่มีข้อมูลจะเป็น NaN)
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    final_df.index = all_months

    # ลบคอลัมน์ year/month ที่ใช้ merge ออกเพื่อให้เหลือเฉพาะชื่อยา
    final_df = final_df.drop(columns=['year', 'month'])

    # เติมค่าว่างด้วยวิธี Linear Interpolation (ลากเส้นตรงระหว่างจุด)
    final_df = final_df.interpolate(method='linear')
    
    # เติมค่ากรณีมีช่องว่างที่หัว/ท้ายตาราง (ที่ interpolate เข้าไม่ถึง)
    final_df = final_df.bfill().ffill() 

    # 2. เลือกกลุ่มยาที่ต้องการวิเคราะห์
    target_drug = 'AMINOGLYCOSIDES, CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        run_mdr_forecasting(series_data, "Pseudomonas aeruginosa")
    else:
        print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
        print("กลุ่มยาที่มีในไฟล์คือ:", final_df.columns.tolist())
=======
    # Merge ข้อมูลเข้ากับตารางวันที่หลัก
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left')
    final_df.index = all_months

    # ลบคอลัมน์ year/month ที่ใช้ merge ออกเพื่อให้เหลือเฉพาะตัวเลข
    final_df = final_df.drop(columns=['year', 'month'])

    # เติมค่าว่างด้วยวิธี Linear Interpolation
    final_df = final_df.interpolate(method='linear')
    
    # เติมค่ากรณีมีช่องว่างที่หัว/ท้ายตาราง
    final_df = final_df.bfill().ffill() 

    # [UPDATE] ระบุชื่อยาที่ต้องการวิเคราะห์
    target_drug = 'cefepime'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        # [UPDATE] เปลี่ยนชื่อให้สื่อความหมายตรงกับไฟล์ข้อมูล S. aureus
        run_mdr_forecasting(series_data, f"Escherichia coli to {target_drug}")
    else:
        print(f"❌ ไม่พบชื่อยา '{target_drug}' ในข้อมูล")
        print(f"รายชื่อยาที่มีทั้งหมดในไฟล์คือ: {list(final_df.columns)}")
>>>>>>> Pammy
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")