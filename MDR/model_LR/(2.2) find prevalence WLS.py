import pandas as pd
import os
import matplotlib.pyplot as plt

INPUT_ORIGINAL = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/data/MDR/AllYears_DrugClass_tested.csv'
OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR'
CLEANED_ABA = os.path.join(OUTPUT_DIR, 'a_baumannii_selected.csv')

def plot_xy_chart(x, y, title, folder, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x.to_numpy(), y.to_numpy(), marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6)
    plt.title(title, fontsize=12, pad=15)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Prevalence (%)", fontsize=10)
    plt.xticks(range(2015, 2025))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def step2_calculate():
    print("⏳ Step 2: Calculating % Prevalence and Saving N for WLS...")
    
    df_all = pd.read_csv(INPUT_ORIGINAL, low_memory=False)
    df_all.columns = df_all.columns.str.lower()
    df_all = df_all[(df_all['x_year'] >= 2015) & (df_all['x_year'] <= 2024)]
    
    df_aba = pd.read_csv(CLEANED_ABA)
    
    for folder in ['Prevalence All Data', 'Prevalence By Ward Type', 'Prevalence By Specimen']:
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
        
    # ==========================================
    # 1. Overall
    # ==========================================
    total_all = df_all.groupby('x_year').size()
    total_aba = df_aba.groupby('x_year').size()
    
    prev_overall = pd.DataFrame({
        'organism_full': 'Acinetobacter baumannii',
        'x_year': total_all.index,
        'n_total': total_all.values,  # <--- ตัวนี้จะเป็น Weight ของภาพรวม
        'n_aba': total_aba.reindex(total_all.index, fill_value=0).values
    })
    prev_overall['prevalence_%'] = ((prev_overall['n_aba'] / prev_overall['n_total']) * 100).round(3)
    prev_overall.to_csv(os.path.join(OUTPUT_DIR, 'Prevalence All Data', 'WLS_overall_prevalence.csv'), index=False)
    
    plot_xy_chart(prev_overall['x_year'], prev_overall['prevalence_%'], 
                  'Overall Prevalence of A. baumannii (2015-2024)', 'Prevalence All Data', 'line_chart_overall')

    # ==========================================
    # 2.1 Ward Type
    # ==========================================
    df_ward_valid = df_aba.dropna(subset=['ward_type']).copy()
    ward_denominator = df_ward_valid.groupby('x_year').size()
    ward_counts = df_ward_valid.groupby(['x_year', 'ward_type']).size().unstack(fill_value=0)
    
    prev_ward = pd.DataFrame(index=ward_denominator.index)
    
    # 🌟 เพิ่มคอลัมน์ n_total_ward เพื่อเอาไปใช้ทำ WLS
    prev_ward['n_total_ward'] = ward_denominator.values 
    
    for w in ['icu', 'in', 'out']:
        if w in ward_counts.columns:
            prev_ward[w] = ((ward_counts[w] / ward_denominator) * 100).round(3)
        else:
            prev_ward[w] = 0.0
            
    prev_ward = prev_ward.reset_index()
    prev_ward.insert(0, 'organism_full', 'Acinetobacter baumannii')
    prev_ward.to_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Ward Type', 'WLS_ward_prevalence.csv'), index=False)

    # ==========================================
    # 2.2 Specimen Types
    # ==========================================
    df_spec_valid = df_aba.dropna(subset=['spec_group']).copy()
    spec_denominator = df_spec_valid.groupby('x_year').size()
    spec_counts = df_spec_valid.groupby(['x_year', 'spec_group']).size().unstack(fill_value=0)
    
    prev_spec = pd.DataFrame(index=spec_denominator.index)
    
    # 🌟 เพิ่มคอลัมน์ n_total_spec เพื่อเอาไปใช้ทำ WLS
    prev_spec['n_total_spec'] = spec_denominator.values
    
    for s in spec_counts.columns:
        prev_spec[s] = ((spec_counts[s] / spec_denominator) * 100).round(3)
        
    prev_spec = prev_spec.reset_index()
    prev_spec.insert(0, 'organism_full', 'Acinetobacter baumannii')
    prev_spec.to_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Specimen', 'WLS_specimen_prevalence.csv'), index=False)

    print("✅ Step 2 Complete: Saved N-total columns for WLS calculation.")

if __name__ == "__main__":
    step2_calculate()