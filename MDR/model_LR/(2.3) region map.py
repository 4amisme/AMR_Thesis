import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

def plot_health_region_map():

    CSV_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/Prevalence All Data/map_data_region_2024.csv'
    GEOJSON_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/Regions_no_province_boundaries.json'

    OUTPUT_DIR = os.path.dirname(CSV_PATH)
    SAVE_PATH = os.path.join(OUTPUT_DIR, 'map_infections_2024.png')

    try:
        df = pd.read_csv(CSV_PATH)
        df['region'] = df['region'].astype(int) 
    except Exception as e:
        print(f"ไม่พบไฟล์ CSV: {e}")
        return

    try:
        gdf = gpd.read_file(GEOJSON_PATH)
        # แปลง HealthRegion เป็น int ให้ตรงกัน
        gdf['HealthRegion'] = gdf['HealthRegion'].astype(int)
    except Exception as e:
        print(f"ไม่สามารถโหลดไฟล์แผนที่ GeoJSON : {e}")
        return

    merged_gdf = gdf.merge(df, left_on='HealthRegion', right_on='region', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    merged_gdf.plot(
        column='n_aba',               
        cmap='OrRd',                  
        linewidth=0.8,                
        edgecolor='black',            
        legend=True,                  
        legend_kwds={
            'label': 'Number of A. baumannii Infections (2024)', 
            'orientation': 'vertical',
        },
        missing_kwds={'color': 'lightgrey'}, 
        ax=ax
    )

    plt.title('Acinetobacter baumannii Infections by Health Region (2024)', fontsize=15, pad=15)
    ax.axis('off') 
    plt.tight_layout()

    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"บันทึกไฟล์ไว้ที่: \n{SAVE_PATH}")

if __name__ == "__main__":
    plot_health_region_map()

def plot_prevalence_map():
    INPUT_ORIGINAL = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/data/MDR/AllYears_DrugClass_tested.csv'
    CLEANED_ABA = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/a_baumannii_selected.csv'
    GEOJSON_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/Regions_no_province_boundaries.json'
    OUTPUT_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/Prevalence All Data'
    
    SAVE_CSV_PATH = os.path.join(OUTPUT_DIR, 'prevalence_region_2024.csv')
    SAVE_MAP_PATH = os.path.join(OUTPUT_DIR, 'map_prevalence_2024.png')

    try:
        # n_total ของปี 2024
        df_all = pd.read_csv(INPUT_ORIGINAL, low_memory=False)
        df_all.columns = df_all.columns.str.lower()
        df_all_2024 = df_all[df_all['x_year'] == 2024]
        total_by_region = df_all_2024.groupby('region').size().reset_index(name='n_total')
        
        # n_aba ของปี 2024
        df_aba = pd.read_csv(CLEANED_ABA)
        df_aba_2024 = df_aba[df_aba['x_year'] == 2024]
        aba_by_region = df_aba_2024.groupby('region').size().reset_index(name='n_aba')
        
        df_prev = pd.merge(total_by_region, aba_by_region, on='region', how='left')
        df_prev['n_aba'] = df_prev['n_aba'].fillna(0)
        df_prev['prevalence_%'] = ((df_prev['n_aba'] / df_prev['n_total']) * 100).round(2)
        
        df_prev.to_csv(SAVE_CSV_PATH, index=False)
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการคำนวณข้อมูล: {e}")
        return
    try:
        gdf = gpd.read_file(GEOJSON_PATH)
        gdf['HealthRegion'] = gdf['HealthRegion'].astype(int)
        df_prev['region'] = df_prev['region'].astype(int)
        
        merged_gdf = gdf.merge(df_prev, left_on='HealthRegion', right_on='region', how='left')
    except Exception as e:
        print(f"เกิดข้อผิดพลาดเกี่ยวกับไฟล์แผนที่ GeoJSON: {e}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    merged_gdf.plot(
        column='prevalence_%',        
        cmap='OrRd',                  
        linewidth=0.8,                
        edgecolor='black',            
        legend=True,                  
        legend_kwds={
            'label': 'A. baumannii Prevalence (%) in 2024', 
            'orientation': 'vertical',
            'shrink': 0.6             
        },
        missing_kwds={'color': 'lightgrey'}, 
        ax=ax
    )

    plt.title('Prevalence of Acinetobacter baumannii by Health Region (2024)', fontsize=15, pad=15)
    ax.axis('off') 
    plt.tight_layout()

    plt.savefig(SAVE_MAP_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"บันทึกไฟล์ไว้ที่:\n{SAVE_MAP_PATH}")

if __name__ == "__main__":
    plot_prevalence_map()