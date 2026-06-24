import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import statsmodels.api as sm

# ตั้งค่า Path หลัก
OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR'

def run_model_and_plot(df, x_col, y_col, title, folder_path, filename, y_max=None):
    """ฟังก์ชันรัน Linear Regression, เซฟ OLS Summary และพลอตกราฟ Trendline"""
    x = df[x_col]
    y = df[y_col]
    
    # คำนวณสถิติ
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    X_sm = sm.add_constant(x)
    model = sm.OLS(y, X_sm)
    results = model.fit()
    
    # สร้างโฟลเดอร์สำหรับเก็บไฟล์ Summary สถิติ (.txt)
    summary_dir = os.path.join(folder_path, 'OLS_Summaries')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{filename}_OLS_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write(results.summary().as_text())
    
    # พลอตกราฟ
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
    
    save_path = os.path.join(folder_path, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return {
        "Category": title, 
        "Slope": slope, 
        "P-value": p_value, 
        "R-squared": r_value**2
    }

def plot_combined_models(df, x_col, y_cols, title, folder_path, filename):
    """ฟังก์ชันสำหรับพลอตกราฟหลายเส้นพร้อม Trendline รวมกันในรูปเดียว"""
    plt.figure(figsize=(10, 6))
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    for idx, col in enumerate(y_cols):
        if col not in df.columns: continue
        x = df[x_col]
        y = df[col]
        c = colors[idx % len(colors)]
        
        # คำนวณเส้นเทรนด์
        slope, intercept, r, p, se = linregress(x, y)
        y_pred = intercept + (slope * x)
        
        # พล็อตข้อมูลจริงและเส้นเทรนด์
        plt.plot(x.to_numpy(), y.to_numpy(), marker='o', linestyle='-', color=c, linewidth=2, label=f'{col.upper()} (Actual)')
        plt.plot(x.to_numpy(), y_pred.to_numpy(), linestyle='--', color=c, alpha=0.6, linewidth=2, label=f'{col.upper()} (Trend: p={p:.3f})')
        
    plt.title(title, fontsize=13, pad=15)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Prevalence (%)", fontsize=10)
    plt.xticks(range(2015, 2025))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(folder_path, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def step3_modeling_loop():
    target_organisms = [
        'Acinetobacter baumannii',
        'Enterococcus faecalis',
        'Enterococcus faecium',
        'Escherichia coli',
        'Klebsiella pneumoniae ',
        'Pseudomonas aeruginosa ',
        'Staphylococcus aureus'
    ]

    for org_name in target_organisms:
        print(f"\n📈 Step 3 Modeling: {org_name}...")
        summary_results = []
        
        # 📂 1. Overall Prevalence
        org_overall_dir = os.path.join(OUTPUT_DIR, 'Prevalence All Data', org_name)
        overall_csv = os.path.join(org_overall_dir, 'overall_prevalence.csv')
        
        if os.path.exists(overall_csv):
            df_overall = pd.read_csv(overall_csv)
            res = run_model_and_plot(df_overall, 'x_year', 'prevalence_%', 
                                     f'Overall {org_name} Prevalence Trend', 
                                     org_overall_dir, 'plot_trend_overall')
            summary_results.append(res)
        
        # 📂 2. Ward Type Prevalence
        org_ward_dir = os.path.join(OUTPUT_DIR, 'Prevalence By Ward Type', org_name)
        ward_csv = os.path.join(org_ward_dir, 'ward_prevalence.csv')
        
        if os.path.exists(ward_csv):
            df_ward = pd.read_csv(ward_csv)
            wards = ['icu', 'in', 'out']
            # หาค่า Max เพื่อตั้งสเกลแกน Y ให้เท่ากันในหมวดนี้
            ward_max = df_ward[wards].max().max() + 5
            
            for w in wards:
                res = run_model_and_plot(df_ward, 'x_year', w, 
                                         f'Prevalence by Ward Type: {w.upper()} ({org_name})', 
                                         org_ward_dir, f'plot_trend_ward_{w}', y_max=ward_max)
                summary_results.append(res)

            plot_combined_models(df_ward, 'x_year', wards, 
                                 f'Combined Prevalence & Trendlines by Ward ({org_name})', 
                                 org_ward_dir, 'plot_ward_combined_trendlines')
            
        # 📂 3. Specimen Prevalence
        org_spec_dir = os.path.join(OUTPUT_DIR, 'Prevalence By Specimen', org_name)
        spec_csv = os.path.join(org_spec_dir, 'specimen_prevalence.csv')
        
        if os.path.exists(spec_csv):
            df_spec = pd.read_csv(spec_csv)
            spec_cols = [col for col in df_spec.columns if col not in ['organism_full', 'x_year']]
            spec_max = df_spec[spec_cols].max().max() + 5
            
            for s in spec_cols:
                safe_filename = str(s).replace('/', '_').replace(' ', '_')
                res = run_model_and_plot(df_spec, 'x_year', s, 
                                         f'Prevalence by Specimen: {s} ({org_name})', 
                                         org_spec_dir, f'plot_trend_spec_{safe_filename}', y_max=spec_max)
                summary_results.append(res)
                
            # พล็อตรวมเฉพาะ Top 3 (ไม่รวม 'other')
            top3_cols = [c for c in spec_cols if c != 'other']
            plot_combined_models(df_spec, 'x_year', top3_cols, 
                                 f'Combined Trendlines by Specimen (Top 3) ({org_name})', 
                                 org_spec_dir, 'plot_spec_combined_trendlines')
        
        # เซฟตารางสรุปค่าสถิติรวมของเชื้อตัวนี้
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            summary_df.to_csv(os.path.join(OUTPUT_DIR, f'summary_models_{org_name.replace(" ", "_")}.csv'), index=False)
            print(f"✅ Modeling complete for {org_name}")

    print("\n🚀 All organisms modeled and plotted successfully!")

if __name__ == "__main__":
    step3_modeling_loop()