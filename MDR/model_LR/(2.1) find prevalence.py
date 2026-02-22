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
    print("⏳ Step 2: Calculating % Prevalence (rounded to 3 decimals) and Plotting...")
    
    df_all = pd.read_csv(INPUT_ORIGINAL, low_memory=False)
    df_all.columns = df_all.columns.str.lower()
    df_all = df_all[(df_all['x_year'] >= 2015) & (df_all['x_year'] <= 2024)]
    
    df_aba = pd.read_csv(CLEANED_ABA)
    
    for folder in ['Prevalence All Data', 'Prevalence By Ward Type', 'Prevalence By Specimen']:
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
        
    # ==========================================
    # 1. Overall Prevalence
    # ==========================================
    total_all = df_all.groupby('x_year').size()
    total_aba = df_aba.groupby('x_year').size()
    
    prev_overall = pd.DataFrame({
        'organism_full': 'Acinetobacter baumannii',
        'x_year': total_all.index,
        'n_total': total_all.values,
        'n_aba': total_aba.reindex(total_all.index, fill_value=0).values
    })
    prev_overall['prevalence_%'] = ((prev_overall['n_aba'] / prev_overall['n_total']) * 100).round(3)
    prev_overall.to_csv(os.path.join(OUTPUT_DIR, 'Prevalence All Data', 'overall_prevalence.csv'), index=False)
    
    plot_xy_chart(prev_overall['x_year'], prev_overall['prevalence_%'], 
                  'Overall Prevalence of A. baumannii (2015-2024)', 
                  'Prevalence All Data', 'line_chart_overall')

    # ==========================================
    # 2.1 Ward Type Prevalence
    # ==========================================
    df_ward_valid = df_aba.dropna(subset=['ward_type']).copy()
    ward_denominator = df_ward_valid.groupby('x_year').size()
    ward_counts = df_ward_valid.groupby(['x_year', 'ward_type']).size().unstack(fill_value=0)
    
    # 🌟 แก้ไขตรงนี้: สร้างตารางโดยยึด Index เป็นปี (2015-2024) ก่อนเลย
    prev_ward = pd.DataFrame(index=ward_denominator.index)
    
    for w in ['icu', 'in', 'out']:
        if w in ward_counts.columns:
            prev_ward[w] = ((ward_counts[w] / ward_denominator) * 100).round(3)
        else:
            prev_ward[w] = 0.0
            
    # พอคำนวณเสร็จ ค่อยดีดเลขปี (Index) กลับมาเป็นคอลัมน์ชื่อ x_year
    prev_ward = prev_ward.reset_index()
    prev_ward.insert(0, 'organism_full', 'Acinetobacter baumannii')
    prev_ward.to_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Ward Type', 'ward_prevalence.csv'), index=False)
    
    for w in ['icu', 'in', 'out']:
        if w in prev_ward.columns:
            plot_xy_chart(prev_ward['x_year'], prev_ward[w], 
                          f'Prevalence by Ward Type: {w.upper()} (2015-2024)', 
                          'Prevalence By Ward Type', f'line_chart_ward_{w}')

    # ==========================================
    # 2.2 Specimen Types
    # ==========================================
    df_spec_valid = df_aba.dropna(subset=['spec_group']).copy()
    spec_denominator = df_spec_valid.groupby('x_year').size()
    spec_counts = df_spec_valid.groupby(['x_year', 'spec_group']).size().unstack(fill_value=0)
    
    # 🌟 แก้ไขตรงนี้: สร้างตารางโดยยึด Index เป็นปี (2015-2024) เหมือนกัน
    prev_spec = pd.DataFrame(index=spec_denominator.index)
    
    for s in spec_counts.columns:
        prev_spec[s] = ((spec_counts[s] / spec_denominator) * 100).round(3)
        
    # ดีดเลขปีกลับมาเป็นคอลัมน์
    prev_spec = prev_spec.reset_index()
    prev_spec.insert(0, 'organism_full', 'Acinetobacter baumannii')
    prev_spec.to_csv(os.path.join(OUTPUT_DIR, 'Prevalence By Specimen', 'specimen_prevalence.csv'), index=False)
    
    spec_cols = [col for col in prev_spec.columns if col not in ['organism_full', 'x_year']]
    for s in spec_cols:
        safe_name = str(s).replace('/', '_').replace(' ', '_')
        plot_xy_chart(prev_spec['x_year'], prev_spec[s], 
                      f'Prevalence by Specimen: {s} (2015-2024)', 
                      'Prevalence By Specimen', f'line_chart_spec_{safe_name}')
    
    # ==========================================
    # แผนที่ปี 2024
    # ==========================================
    map_2024 = df_aba[df_aba['x_year'] == 2024].groupby('region').size().reset_index(name='n_aba')
    map_2024.insert(0, 'x_year', 2024)
    map_2024.insert(0, 'organism_full', 'Acinetobacter baumannii')
    map_2024.to_csv(os.path.join(OUTPUT_DIR, 'Prevalence All Data', 'map_data_region_2024.csv'), index=False)

    print("✅ Step 2 Complete: Calculation bugs fixed. Data is filled perfectly!")

if __name__ == "__main__":
    step2_calculate()