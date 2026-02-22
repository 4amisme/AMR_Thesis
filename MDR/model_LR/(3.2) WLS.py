import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm

OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR'

def run_model_and_plot(df, x_col, y_col, weight_col, title, folder, filename, y_max=None):
    x = df[x_col]
    y = df[y_col]
    w = df[weight_col]
    
    # ==========================================
    # 1. รัน OLS (แบบธรรมดา)
    # ==========================================
    slope_ols, intercept_ols, r_value_ols, p_value_ols, std_err_ols = linregress(x, y)
    
    # ==========================================
    # 2. รัน WLS (ถ่วงน้ำหนักด้วย N)
    # ==========================================
    X_sm = sm.add_constant(x) # เติมค่าคงที่เพื่อให้สมการสมบูรณ์ (Y = mX + c)
    wls_model = sm.WLS(y, X_sm, weights=w)
    wls_results = wls_model.fit()
    
    slope_wls = wls_results.params[x_col]
    intercept_wls = wls_results.params['const']
    p_value_wls = wls_results.pvalues[x_col]
    r_squared_wls = wls_results.rsquared
    
    # ==========================================
    # 3. วาดกราฟเปรียบเทียบ
    # ==========================================
    plt.figure(figsize=(9, 6))
    
    # พล็อต Data Point และเส้นกราฟจริง
    plt.plot(x.to_numpy(), y.to_numpy(), marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Actual Prevalence (%)')
    
    # พล็อตเส้น Trendline OLS (สีแดง)
    y_pred_ols = intercept_ols + (slope_ols * x)
    plt.plot(x.to_numpy(), y_pred_ols.to_numpy(), color='red', linestyle='--', linewidth=2, label='OLS Trendline')
    
    # พล็อตเส้น Trendline WLS (สีเขียว)
    y_pred_wls = intercept_wls + (slope_wls * x)
    plt.plot(x.to_numpy(), y_pred_wls.to_numpy(), color='green', linestyle='-.', linewidth=2, label='WLS Trendline (Weighted by N)')
    
    # ตกแต่งกราฟให้โชว์ค่าสถิติทั้ง 2 แบบบนหัวกราฟเลย
    sign_ols = "+" if slope_ols > 0 else ""
    sign_wls = "+" if slope_wls > 0 else ""
    
    title_text = (f"{title}\n"
                  f"OLS: Slope = {sign_ols}{slope_ols:.4f} | p-value = {p_value_ols:.4f}\n"
                  f"WLS: Slope = {sign_wls}{slope_wls:.4f} | p-value = {p_value_wls:.4f}")
    
    plt.title(title_text, fontsize=11, pad=15)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Prevalence (%)", fontsize=10)
    plt.xticks(range(2015, 2025))
    
    if y_max is not None:
        plt.ylim(0, y_max)
        
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # คืนค่าสถิติทั้งหมดลงตารางสรุปผล
    return {
        "Category": title, 
        "OLS_Slope": slope_ols, "OLS_P-value": p_value_ols, "OLS_R2": r_value_ols**2,
        "WLS_Slope": slope_wls, "WLS_P-value": p_value_wls, "WLS_R2": r_squared_wls
    }

def step3_modeling():
    print("⏳ Step 3: Running OLS vs WLS Regression & Generating Plots...")
    summary_results = []
    
    # 1. Overall
    df_overall = pd.read_csv(os.path.join(OUTPUT_DIR, 'Prevalence All Data', 'WLS_overall_prevalence.csv'))
    res = run_model_and_plot(df_overall, 'x_year', 'prevalence_%', 'n_total',
                             'Overall A. baumannii Prevalence (2015-2024)', 
                             'Prevalence All Data', 'plot_overall_OLS_vs_WLS')
    summary_results.append(res)
    
    # 2. Ward
    df_ward = pd.read_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Ward Type', 'WLS_ward_prevalence.csv'))
    ward_max = df_ward[['icu', 'in', 'out']].max().max() + 5
    for w in ['icu', 'in', 'out']:
        if w in df_ward.columns:
            res = run_model_and_plot(df_ward, 'x_year', w, 'n_total_ward',
                                     f'Prevalence by Ward Type: {w.upper()} (2015-2024)', 
                                     'Prevalence By Ward Type', f'plot_ward_{w}_OLS_vs_WLS', y_max=ward_max)
            summary_results.append(res)
            
    # 3. Specimen
    df_spec = pd.read_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Specimen', 'WLS_specimen_prevalence.csv'))
    spec_cols = [col for col in df_spec.columns if col not in ['organism_full', 'x_year', 'n_total_spec']]
    spec_max = df_spec[spec_cols].max().max() + 5
    
    for s in spec_cols:
        safe_filename = str(s).replace('/', '_').replace(' ', '_')
        res = run_model_and_plot(df_spec, 'x_year', s, 'n_total_spec',
                                 f'Prevalence by Specimen: {s} (2015-2024)', 
                                 'Prevalence By Specimen', f'plot_spec_{safe_filename}_OLS_vs_WLS', y_max=spec_max)
        summary_results.append(res)
        
    # บันทึกตารางสรุปที่มี OLS และ WLS คู่กัน
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'all_models_summary_OLS_vs_WLS.csv'), index=False)
    
    print("✅ Step 3 Complete: OLS vs WLS models ran successfully!")

if __name__ == "__main__":
    step3_modeling()