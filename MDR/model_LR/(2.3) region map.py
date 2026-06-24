import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. ตั้งค่า Path หลัก
# ==========================================
INPUT_ORIGINAL = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/data/MDR/AllYears_DrugClass_tested.csv'
GEOJSON_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/Regions_no_province_boundaries.json'
BASE_OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR'

def run_organism_maps():
    # รายชื่อเชื้อทั้งหมดที่คุณแปมต้องการรัน
    target_organisms = [
        'Klebsiella pneumoniae ',
        'Pseudomonas aeruginosa '
    ]

    print("⏳ Loading master data (Total 2024)...")
    try:
        # โหลดข้อมูลดิบครั้งเดียวเพื่อหา n_total ของปี 2024 ในแต่ละเขต
        df_all = pd.read_csv(INPUT_ORIGINAL, low_memory=False)
        df_all.columns = df_all.columns.str.lower()
        df_all_2024 = df_all[df_all['x_year'] == 2024].copy()
        total_by_region = df_all_2024.groupby('region').size().reset_index(name='n_total')
        total_by_region['region'] = total_by_region['region'].astype(int)
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดไฟล์ Original ได้: {e}")
        return

    print("⏳ Loading GeoJSON map...")
    try:
        gdf = gpd.read_file(GEOJSON_PATH)
        gdf['HealthRegion'] = gdf['HealthRegion'].astype(int)
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดไฟล์ GeoJSON ได้: {e}")
        return

    # ==========================================
    # 2. เริ่ม Loop รันทีละเชื้อ
    # ==========================================
    for org_name in target_organisms:
        print(f"\n🌍 Processing Maps for: {org_name}...")
        
        # สร้างชื่อไฟล์ Cleaned (เช่น a_baumannii_selected.csv)
        clean_file_name = org_name.lower().replace(' ', '_').replace('.', '') + '_selected.csv'
        cleaned_path = os.path.join(BASE_OUTPUT_DIR, clean_file_name)
        
        # สร้างโฟลเดอร์ผลลัพธ์แยกตามเชื้อ
        org_output_dir = os.path.join(BASE_OUTPUT_DIR, 'Prevalence All Data', org_name)
        os.makedirs(org_output_dir, exist_ok=True)

        if not os.path.exists(cleaned_path):
            print(f"⚠️ ไม่พบไฟล์ {cleaned_path}, ข้ามเชื้อนี้...")
            continue

        try:
            # 2.1 คำนวณ Prevalence 2024
            df_org = pd.read_csv(cleaned_path)
            df_org_2024 = df_org[df_org['x_year'] == 2024].copy()
            org_by_region = df_org_2024.groupby('region').size().reset_index(name='n_org')
            org_by_region['region'] = org_by_region['region'].astype(int)

            df_prev = pd.merge(total_by_region, org_by_region, on='region', how='left')
            df_prev['n_org'] = df_prev['n_org'].fillna(0)
            df_prev['prevalence_%'] = ((df_prev['n_org'] / df_prev['n_total']) * 100).round(2)
            
            # บันทึก CSV ประจำเชื้อ
            df_prev.to_csv(os.path.join(org_output_dir, f'prevalence_region_2024_{org_name}.csv'), index=False)

            # 2.2 รวมข้อมูลเข้ากับแผนที่
            merged_gdf = gdf.merge(df_prev, left_on='HealthRegion', right_on='region', how='left')

            # ==========================================
            # 3. พล็อตแผนที่ 2 แบบ (Infections และ Prevalence)
            # ==========================================
            
            # --- แผนที่ 1: จำนวนการติดเชื้อ (Infections) ---
            fig, ax = plt.subplots(1, 1, figsize=(10, 12))
            merged_gdf.plot(column='n_org', cmap='OrRd', linewidth=0.8, edgecolor='black', legend=True,
                            legend_kwds={'label': f'Number of {org_name} Infections (2024)', 'orientation': 'vertical', 'shrink': 0.6},
                            missing_kwds={'color': 'lightgrey'}, ax=ax)
            ax.set_title(f'{org_name} Infections by Health Region (2024)', fontsize=14, pad=10)
            ax.axis('off')
            plt.savefig(os.path.join(org_output_dir, f'map_infections_2024.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # --- แผนที่ 2: อัตราความชุก (Prevalence) ---
            fig, ax = plt.subplots(1, 1, figsize=(10, 12))
            merged_gdf.plot(column='prevalence_%', cmap='YlOrRd', linewidth=0.8, edgecolor='black', legend=True,
                            legend_kwds={'label': f'{org_name} Prevalence (%) in 2024', 'orientation': 'vertical', 'shrink': 0.6},
                            missing_kwds={'color': 'lightgrey'}, ax=ax)
            ax.set_title(f'Prevalence of {org_name} by Health Region (2024)', fontsize=14, pad=10)
            ax.axis('off')
            plt.savefig(os.path.join(org_output_dir, f'map_prevalence_2024.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ บันทึกแผนที่ของ {org_name} เรียบร้อยที่โฟลเดอร์: {org_output_dir}")

        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดกับเชื้อ {org_name}: {e}")

if __name__ == "__main__":
    run_organism_maps()