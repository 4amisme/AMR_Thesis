import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# กำหนด Path หลัก
OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR'

def run_model_and_plot(df, x_col, y_col, title, folder, filename, y_max=None):
    """ฟังก์ชันคำนวณ Linear Regression และวาดกราฟ XY Chart"""
    x = df[x_col]
    y = df[y_col]
    
    # 1. รัน Linear Regression หาค่าทางสถิติ
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # 2. เตรียมวาดกราฟ
    plt.figure(figsize=(8, 5))
    
    # พล็อต Data Point และเส้นกราฟจริง (ป้องกัน Error ด้วย .to_numpy())
    plt.plot(x.to_numpy(), y.to_numpy(), marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Actual Prevalence (%)')
    
    # พล็อตเส้นแนวโน้ม (Trendline)
    sns.regplot(x=x.to_numpy(), y=y.to_numpy(), scatter=False, color='red', line_kws={'linestyle': '--', 'label': 'Trendline'})
    
    # ตกแต่งกราฟและใส่ค่าทางสถิติ
    sign = "+" if slope > 0 else ""
    plt.title(f"{title}\nSlope: {sign}{slope:.4f} | p-value: {p_value:.4f}", fontsize=12, pad=15)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Prevalence (%)", fontsize=10)
    
    # บังคับแกน X เป็น 2015-2024
    plt.xticks(range(2015, 2025))
    
    # 🌟 ถ้าระบุ y_max มา ให้ล็อคสเกลแกน Y ให้เท่ากัน
    if y_max is not None:
        plt.ylim(0, y_max)
        
    # เปิด Grid และ Legend
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 3. เซฟรูปภาพลงโฟลเดอร์ที่กำหนด
    save_path = os.path.join(OUTPUT_DIR, folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # ส่งคืนค่าเพื่อเก็บลงตารางสรุป
    return {
        "Category": title, 
        "Slope": slope, 
        "P-value": p_value, 
        "R-squared": r_value**2
    }

def step3_modeling():
    print("⏳ Step 3: Running Linear Regression & Generating Plots...")
    summary_results = []
    
    # ==========================================
    # 1. โมเดลภาพรวม (Overall)
    # ==========================================
    df_overall = pd.read_csv(os.path.join(OUTPUT_DIR, 'Prevalence All Data', 'overall_prevalence.csv'))
    # ภาพรวมมีรูปเดียว ไม่ต้องล็อคสเกลแกน Y
    res = run_model_and_plot(df_overall, 'x_year', 'prevalence_%', 
                             'Overall A. baumannii Prevalence (2015-2024)', 
                             'Prevalence All Data', 'plot_overall')
    summary_results.append(res)
    
    # ==========================================
    # 2. โมเดลแยกตามแผนก (Ward Type)
    # ==========================================
    df_ward = pd.read_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Ward Type', 'ward_prevalence.csv'))
    
    # หาค่าเปอร์เซ็นต์ที่สูงที่สุดของแผนกทั้งหมด แล้วบวกเผื่อไปอีก 5% เพื่อใช้ล็อคแกน Y
    ward_max = df_ward[['icu', 'in', 'out']].max().max() + 5
    
    for w in ['icu', 'in', 'out']:
        if w in df_ward.columns:
            res = run_model_and_plot(df_ward, 'x_year', w, 
                                     f'Prevalence by Ward Type: {w.upper()} (2015-2024)', 
                                     'Prevalence By Ward Type', f'plot_ward_{w}', y_max=ward_max)
            summary_results.append(res)
            
    # ==========================================
    # 3. โมเดลแยกตามสิ่งส่งตรวจ (Specimen)
    # ==========================================
    df_spec = pd.read_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Specimen', 'specimen_prevalence.csv'))
    spec_cols = [col for col in df_spec.columns if col not in ['organism_full', 'x_year']]
    
    # หาค่าเปอร์เซ็นต์ที่สูงที่สุดของสิ่งส่งตรวจทั้งหมด แล้วบวกเผื่อไปอีก 5% เพื่อใช้ล็อคแกน Y
    spec_max = df_spec[spec_cols].max().max() + 5
    
    for s in spec_cols:
        safe_filename = str(s).replace('/', '_').replace(' ', '_')
        res = run_model_and_plot(df_spec, 'x_year', s, 
                                 f'Prevalence by Specimen: {s} (2015-2024)', 
                                 'Prevalence By Specimen', f'plot_spec_{safe_filename}', y_max=spec_max)
        summary_results.append(res)
        
    # ==========================================
    # บันทึกตารางสรุปผลโมเดลทั้งหมด
    # ==========================================
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'all_models_summary.csv'), index=False)
    
    print("✅ Step 3 Complete: All regression models ran and charts plotted!")
    print(f"📊 สรุปค่า Slope และ P-value ทั้งหมดถูกเก็บไว้ที่: {os.path.join(OUTPUT_DIR, 'all_models_summary.csv')}")

if __name__ == "__main__":
    step3_modeling()