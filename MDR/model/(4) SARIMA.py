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
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else 0
    return round(rmse, 4), round(wape, 4)

# ==========================================
# 2. ฟังก์ชันหลักสำหรับวิเคราะห์และทำนาย
# ==========================================

def run_mdr_forecasting(series, target_drug_name, forecast_months=60):
    print(f"\n{'='*50}")
    print(f"Analyzing: {target_drug_name}")
    print(f"{'='*50}")

    # --- [A] การแบ่งข้อมูล (80/20 สำหรับ Validation) ---
    n = len(series)                         # 1. นับจำนวนเดือนทั้งหมดที่มีในข้อมูล
    train_size = int(n * 0.80)              # 2. คำนวณหาจุดตัดที่ 80% ของจำนวนเดือนทั้งหมด
    train_data = series.iloc[:train_size]   # 3. ตัดเอาข้อมูลตั้งแต่เดือนแรก จนถึงจุดตัด (อดีต)
    test_data = series.iloc[train_size:]    # 4. ตัดเอาข้อมูลตั้งแต่จุดตัด จนถึงเดือนล่าสุด (ปัจจุบัน)

    # --- [B] วิเคราะห์ ACF / PACF ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(series, lags=24, ax=ax1)
    ax1.set_title(f'ACF: {target_drug_name}')
    plot_pacf(series, lags=24, ax=ax2)
    ax2.set_title(f'PACF: {target_drug_name}')
    plt.show()

    # --- [C] Parameter Tuning ด้วย auto_arima ---
    print("Finding best SARIMA parameters (Stepwise Search)...")
    
    # ใช้ train_data ในการหาพารามิเตอร์ เพื่อไม่ให้โมเดลเห็นข้อมูล 20% สุดท้าย
    stepwise_model = auto_arima(
        train_data, 
        start_p=0, start_q=0,    # ลองเริ่มสุ่มที่ค่า 1
        max_p=5, max_q=5,        # ค่าสูงสุดที่โมเดลจะลองสุ่ม
        m=12,                    # ข้อมูลรายเดือน (Seasonality = 12), ข้อมูลเราวนลูปทุก 12 เดือน
        start_P=0, 
        seasonal=True,           # เปิดโหมดวิเคราะห์ฤดูกาล
        d=None,                  # ให้โมเดลหาค่า d (differencing) ที่เหมาะสมเอง
        D=1, 
        trace=True,              # แสดงขั้นตอนการหาค่า
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True            # ให้รันเฉพาะค่าที่มีแนวโน้มว่าจะดี
    )

    best_order = stepwise_model.order
    best_seasonal = stepwise_model.seasonal_order
    
    print(f"\n>>> Best Order: {best_order}")
    print(f">>> Best Seasonal Order: {best_seasonal}")
    print(f">>> Best AIC: {stepwise_model.aic():.2f}")

    # --- [D] Model Training & Forecasting ---
    
    print("\n--- 1. Evaluating Model Performance (Train/Test) ---")
    # เทรนโมเดลสำหรับทดสอบด้วยข้อมูล Train Data (80%) 
    model_eval = SARIMAX(train_data, 
                         order=best_order, 
                         seasonal_order=best_seasonal,
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
    
    # ทำนายช่วงเวลาของ Test Data เพื่อวัดความแม่นยำ
    test_pred_sarima = model_eval.forecast(steps=len(test_data))
    rmse, wape = calculate_metrics(test_data, test_pred_sarima)
    print(f"Evaluation on Test Set -> RMSE: {rmse}, WAPE: {wape}%")

    print("\n--- 2. Forecasting Real Future (100% Data) ---")
    # เทรนโมเดลตัวจริงด้วยข้อมูลทั้งหมด (Series) เพื่อความแม่นยำสูงสุดก่อนทำนายอนาคต
    final_model = SARIMAX(series, 
                          order=best_order, 
                          seasonal_order=best_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
    
    # ทำนายอนาคตล่วงหน้า 5 ปี (Forecast)
    forecast_sarima = final_model.forecast(steps=forecast_months)

    # --- [E] การพล็อตแสดงผล ---
    
    plt.figure(figsize=(12, 6))
    
    # พล็อตข้อมูลจริง
    plt.plot(series.index, series.values, 
             color='#377eb8', marker='o', markersize=4, label='Actual Data', linewidth=1.5)
    
    # พล็อตเส้นเชื่อมและ Forecast (สีแดง)
    # รวมจุดสุดท้ายของข้อมูลจริงเข้ากับจุดแรกของ Forecast เพื่อให้เส้นเชื่อมต่อกัน
    forecast_idx = pd.date_range(start=series.index[-1], periods=forecast_months+1, freq='MS')
    forecast_val = np.concatenate([[series.values[-1]], forecast_sarima.values])
    
    plt.plot(forecast_idx, forecast_val, 
             color='#e41a1c', marker='o', markersize=4, linestyle='--', label='Forecast (Next 5 years)', linewidth=1.5)

    # ตกแต่งกราฟ
    plt.title(f'MDR Pattern Prediction: {target_drug_name}', fontsize=13, pad=15)
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

# ปรับ Path ตามที่อยู่ไฟล์จริง
file_path = os.path.join("MDR", "model", "a_baumannii_ur.csv") 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    # 1. เตรียมข้อมูล Wide Format
    pivot_df = df.pivot_table(index=['year', 'month'], columns='Resistant_Drug_Classes', values='percentage')
    
    # สร้าง Index วันที่ให้สมบูรณ์ (2015-2024)
    all_months = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    full_idx = pd.DataFrame({'year': all_months.year, 'month': all_months.month})
    
    final_df = pd.merge(full_idx, pivot_df.reset_index(), on=['year', 'month'], how='left').fillna(0)
    final_df.index = all_months

    # 2. เลือกกลุ่มยาที่ต้องการวิเคราะห์ (ตัวอย่างกลุ่มที่ดื้อหลายชนิด)
    target_drug = 'CARBAPENEMS, CEPHEMS, FLUOROQUINOLONES, β-LACTAM COMBINATION AGENTS'

    if target_drug in final_df.columns:
        series_data = final_df[target_drug]
        run_mdr_forecasting(series_data, "Acinetobacter baumannii")
    else:
        print(f"ไม่พบกลุ่มยาในข้อมูล: {target_drug}")
        print("กลุ่มยาที่มีในไฟล์คือ:", final_df.columns.tolist())
else:
    print(f"ไม่พบไฟล์ข้อมูลที่: {file_path}")