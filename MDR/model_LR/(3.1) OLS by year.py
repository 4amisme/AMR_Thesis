import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import statsmodels.api as sm

# กำหนด Path หลัก
OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR'

def run_model_and_plot(df, x_col, y_col, title, folder, filename, y_max=None):
    """ฟังก์ชันคำนวณ Linear Regression, วาดกราฟ และบันทึก OLS Summary"""
    x = df[x_col]
    y = df[y_col]
    
    # 1. รัน Linear Regression (SciPy) สำหรับวาดกราฟเบื้องต้น
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # 🌟 2. รัน OLS ด้วย statsmodels เพื่อสกัดตาราง Summary ฉบับเต็ม
    X_sm = sm.add_constant(x)
    model = sm.OLS(y, X_sm)
    results = model.fit()
    
    # สร้างโฟลเดอร์สำหรับเก็บตาราง Summary ถ้าย้งไม่มี
    summary_dir = os.path.join(OUTPUT_DIR, folder, 'OLS_Summaries')
    os.makedirs(summary_dir, exist_ok=True)
    
    # บันทึกตาราง Summary ออกเป็นไฟล์ .txt
    summary_path = os.path.join(summary_dir, f"{filename}_OLS_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(results.summary().as_text())
    
    # 3. เตรียมวาดกราฟ
    plt.figure(figsize=(8, 5))
    plt.plot(x.to_numpy(), y.to_numpy(), marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Actual Prevalence (%)')
    sns.regplot(x=x.to_numpy(), y=y.to_numpy(), scatter=False, color='red', line_kws={'linestyle': '--', 'label': 'Trendline'})
    
    sign = "+" if slope > 0 else ""
    plt.title(f"{title}\nSlope: {sign}{slope:.4f} | p-value: {p_value:.4f}", fontsize=12, pad=15)
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
    
    return {
        "Category": title, 
        "Slope": slope, 
        "P-value": p_value, 
        "R-squared": r_value**2
    }

def plot_combined_models(df, x_col, y_cols, title, folder, filename):
    """ฟังก์ชันสำหรับพล็อตรวมหลายเส้น พร้อม Trendline ในรูปเดียว"""
    plt.figure(figsize=(10, 6))
    colors = ['#d62728', '#1f77b4', '#ff7f0e'] # แดง(ICU/Blood), ฟ้า(IN/Sputum), ส้ม(OUT/Urine)
    
    for idx, col in enumerate(y_cols):
        if col not in df.columns: continue
        x = df[x_col]
        y = df[col]
        c = colors[idx % len(colors)]
        
        # รันหา Trendline
        slope, intercept, r, p, se = linregress(x, y)
        y_pred = intercept + (slope * x)
        
        # วาดเส้นข้อมูลจริง (ทึบ)
        plt.plot(x.to_numpy(), y.to_numpy(), marker='o', linestyle='-', color=c, linewidth=2, label=f'{col.upper()} (Actual)')
        # วาดเส้น Trendline (ประ)
        plt.plot(x.to_numpy(), y_pred.to_numpy(), linestyle='--', color=c, alpha=0.7, linewidth=2, label=f'{col.upper()} (Trend: p={p:.3f})')
        
    plt.title(title, fontsize=13, pad=15)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Prevalence (%)", fontsize=10)
    plt.xticks(range(2015, 2025))
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # ย้าย Legend ไปไว้ด้านนอกเพื่อไม่ให้บังเส้นกราฟ
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def step3_modeling():
    print("⏳ Step 3: Running Linear Regression, Generating OLS Summaries & Combined Plots...")
    summary_results = []
    
    # 1. Overall
    df_overall = pd.read_csv(os.path.join(OUTPUT_DIR, 'Prevalence All Data', 'overall_prevalence.csv'))
    res = run_model_and_plot(df_overall, 'x_year', 'prevalence_%', 
                             'Overall A. baumannii Prevalence (2015-2024)', 
                             'Prevalence All Data', 'plot_overall')
    summary_results.append(res)
    
    # 2. Ward Type
    df_ward = pd.read_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Ward Type', 'ward_prevalence.csv'))
    ward_max = df_ward[['icu', 'in', 'out']].max().max() + 5
    for w in ['icu', 'in', 'out']:
        if w in df_ward.columns:
            res = run_model_and_plot(df_ward, 'x_year', w, 
                                     f'Prevalence by Ward Type: {w.upper()} (2015-2024)', 
                                     'Prevalence By Ward Type', f'plot_ward_{w}', y_max=ward_max)
            summary_results.append(res)
            
    # 🌟 วาดกราฟรวมพร้อม Trendline (Ward)
    plot_combined_models(df_ward, 'x_year', ['icu', 'in', 'out'], 
                         'Combined Prevalence and Trendlines by Ward Type', 
                         'Prevalence By Ward Type', 'plot_ward_combined_trendlines')
            
    # 3. Specimen
    df_spec = pd.read_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Specimen', 'specimen_prevalence.csv'))
    spec_cols = [col for col in df_spec.columns if col not in ['organism_full', 'x_year']]
    spec_max = df_spec[spec_cols].max().max() + 5
    for s in spec_cols:
        safe_filename = str(s).replace('/', '_').replace(' ', '_')
        res = run_model_and_plot(df_spec, 'x_year', s, 
                                 f'Prevalence by Specimen: {s} (2015-2024)', 
                                 'Prevalence By Specimen', f'plot_spec_{safe_filename}', y_max=spec_max)
        summary_results.append(res)
        
    # 🌟 วาดกราฟรวมพร้อม Trendline (Specimen Top 3 ไม่เอา Other)
    plot_combined_models(df_spec, 'x_year', ['Blood', 'Sputum', 'Urine'], 
                         'Combined Prevalence and Trendlines by Specimen (Top 3)', 
                         'Prevalence By Specimen', 'plot_spec_combined_top3_trendlines')
        
    # บันทึกตารางสรุป
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'all_models_summary.csv'), index=False)
    
    print("✅ Step 3 Complete!")
    print("📝 ตาราง OLS Summary")

if __name__ == "__main__":
    step3_modeling()