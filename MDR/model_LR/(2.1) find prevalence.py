import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

INPUT_ORIGINAL = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/data/MDR/AllYears_DrugClass_tested.csv'
OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR'

def plot_xy_chart(x, y, title, folder, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x.to_numpy(), y.to_numpy(), marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6)
    plt.title(title, fontsize=12, pad=15)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Prevalence (%)", fontsize=10)
    plt.xticks(range(2015, 2025))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_combined_xy(df, x_col, y_cols, title, folder, filename):
    plt.figure(figsize=(9, 6))
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    for idx, col in enumerate(y_cols):
        if col in df.columns:
            c = colors[idx % len(colors)]
            plt.plot(df[x_col].to_numpy(), df[col].to_numpy(), marker='o', linestyle='-', color=c, linewidth=2, markersize=6, label=col.upper())
            
    plt.title(title, fontsize=13, pad=15)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Prevalence (%)", fontsize=10)
    plt.xticks(range(2015, 2025))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='Category', loc='upper right')
    plt.tight_layout()
    save_path = os.path.join(folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def step2_calculate_loop():
    print("⏳ Loading master data...")
    df_all_master = pd.read_csv(INPUT_ORIGINAL, low_memory=False)
    df_all_master.columns = df_all_master.columns.str.lower()
    df_all_master = df_all_master[(df_all_master['x_year'] >= 2015) & (df_all_master['x_year'] <= 2024)]
    
    total_all_global = df_all_master.groupby('x_year').size()

    target_organisms = [
        'Acinetobacter baumannii',
        'Enterococcus faecalis',
        'Staphylococcus aureus',
        'Enterococcus faecium',
        'Escherichia coli',
        'Klebsiella pneumoniae ',
        'Pseudomonas aeruginosa '
    ]

    for org_name in target_organisms:
        print(f"\n📊 Step 2 Processing: {org_name}...")
        
        # 1. โหลดไฟล์ Cleaned ของเชื้อนั้นๆ
        clean_name = org_name.lower().replace(' ', '_').replace('.', '') + '_selected.csv'
        clean_path = os.path.join(OUTPUT_DIR, clean_name)
        
        if not os.path.exists(clean_path):
            print(f"⚠️ File not found: {clean_path}, skipping.")
            continue
            
        df_org = pd.read_csv(clean_path)

        # 2. สร้าง Folder แยกตามเชื้อข้างในหมวดหมู่หลัก
        base_folders = ['Prevalence All Data', 'Prevalence By Ward Type', 'Prevalence By Specimen']
        path_map = {}
        for bf in base_folders:
            target_path = os.path.join(OUTPUT_DIR, bf, org_name)
            os.makedirs(target_path, exist_ok=True)
            path_map[bf] = target_path

        # --- 2.1 Overall Prevalence ---
        total_org = df_org.groupby('x_year').size()
        prev_overall = pd.DataFrame({
            'organism_full': org_name,
            'x_year': total_all_global.index,
            'n_total': total_all_global.values,
            'n_org': total_org.reindex(total_all_global.index, fill_value=0).values
        })
        prev_overall['prevalence_%'] = ((prev_overall['n_org'] / prev_overall['n_total']) * 100).round(3)
        prev_overall.to_csv(os.path.join(path_map['Prevalence All Data'], 'overall_prevalence.csv'), index=False)
        
        plot_xy_chart(prev_overall['x_year'], prev_overall['prevalence_%'], 
                      f'Overall Prevalence of {org_name} (2015-2024)', 
                      path_map['Prevalence All Data'], 'line_chart_overall')

        # --- 2.2 Ward Type Prevalence ---
        df_ward_valid = df_org.dropna(subset=['ward_type']).copy()
        ward_denominator = df_ward_valid.groupby('x_year').size()
        ward_counts = df_ward_valid.groupby(['x_year', 'ward_type']).size().unstack(fill_value=0)
        
        prev_ward = pd.DataFrame(index=ward_denominator.index)
        wards = ['icu', 'in', 'out']
        for w in wards:
            if w in ward_counts.columns:
                prev_ward[w] = ((ward_counts[w] / ward_denominator) * 100).round(3)
            else:
                prev_ward[w] = 0.0
                
        prev_ward = prev_ward.reset_index()
        prev_ward.insert(0, 'organism_full', org_name)
        prev_ward.to_csv(os.path.join(path_map['Prevalence By Ward Type'], 'ward_prevalence.csv'), index=False)
        
        for w in wards:
            plot_xy_chart(prev_ward['x_year'], prev_ward[w], 
                          f'Prevalence by Ward: {w.upper()} ({org_name})', 
                          path_map['Prevalence By Ward Type'], f'line_chart_ward_{w}')
                         
        plot_combined_xy(prev_ward, 'x_year', wards, 
                         f'Combined Ward Prevalence ({org_name})', 
                         path_map['Prevalence By Ward Type'], 'line_chart_ward_combined')

        # --- 2.3 Specimen Type Prevalence ---
        df_spec_valid = df_org.dropna(subset=['spec_group']).copy()
        spec_denominator = df_spec_valid.groupby('x_year').size()
        spec_counts = df_spec_valid.groupby(['x_year', 'spec_group']).size().unstack(fill_value=0)
        
        prev_spec = pd.DataFrame(index=spec_denominator.index)
        for s in spec_counts.columns:
            prev_spec[s] = ((spec_counts[s] / spec_denominator) * 100).round(3)
            
        prev_spec = prev_spec.reset_index()
        prev_spec.insert(0, 'organism_full', org_name)
        prev_spec.to_csv(os.path.join(path_map['Prevalence By Specimen'], 'specimen_prevalence.csv'), index=False)
        
        # พล็อตรวมเฉพาะ Top 3 ที่หาได้จริงของเชื้อนั้นๆ
        top3_cols = [c for c in spec_counts.columns if c != 'other']
        
        for s in spec_counts.columns:
            safe_name = str(s).replace('/', '_').replace(' ', '_')
            plot_xy_chart(prev_spec['x_year'], prev_spec[s], 
                          f'Prevalence by Specimen: {s} ({org_name})', 
                          path_map['Prevalence By Specimen'], f'line_chart_spec_{safe_name}')
                          
        plot_combined_xy(prev_spec, 'x_year', top3_cols, 
                         f'Combined Specimen Prevalence (Top 3) ({org_name})', 
                         path_map['Prevalence By Specimen'], 'line_chart_spec_combined')

        # --- 2.4 Map Data 2024 ---
        map_2024 = df_org[df_org['x_year'] == 2024].groupby('region').size().reset_index(name='n_org')
        map_2024.insert(0, 'x_year', 2024)
        map_2024.insert(0, 'organism_full', org_name)
        map_2024.to_csv(os.path.join(path_map['Prevalence All Data'], 'map_data_region_2024.csv'), index=False)

    print("\n🚀 Step 2 Loop Complete: All charts and CSVs saved.")

if __name__ == "__main__":
    step2_calculate_loop()