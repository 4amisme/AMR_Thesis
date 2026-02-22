import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

def plot_health_region_map():
    print("⏳ กำลังสร้างแผนที่จำนวนการติดเชื้อ A. baumannii ปี 2024...")

    CSV_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/Prevalence All Data/map_data_region_2024.csv'
    GEOJSON_PATH = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/Regions_no_province_boundaries.json'

    OUTPUT_DIR = os.path.dirname(CSV_PATH)
    SAVE_PATH = os.path.join(OUTPUT_DIR, 'map_infections_2024.png')

    try:
        df = pd.read_csv(CSV_PATH)
        # แปลงเป็น int เพื่อให้เชื่อมกับแผนที่ได้ตรงกัน
        df['region'] = df['region'].astype(int) 
    except Exception as e:
        print(f"❌ ไม่พบไฟล์ หรือเกิดข้อผิดพลาดในการโหลด CSV: {e}")
        return

    try:
        gdf = gpd.read_file(GEOJSON_PATH)
        # แปลง HealthRegion เป็น int ให้ตรงกัน
        gdf['HealthRegion'] = gdf['HealthRegion'].astype(int)
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดไฟล์แผนที่ GeoJSON ได้: {e}")
        return

    merged_gdf = gdf.merge(df, left_on='HealthRegion', right_on='region', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # วาดแผนที่ไล่สีตามคอลัมน์ n_aba
    merged_gdf.plot(
        column='n_aba',               # คอลัมน์ที่ใช้กำหนดระดับสี
        cmap='OrRd',                  # โทนสี ขาว-ส้ม-แดง (Orange-Red)
        linewidth=0.8,                # ความหนาของเส้นขอบเขต
        edgecolor='black',            # สีเส้นแบ่งเขตสุขภาพ
        legend=True,                  # เปิดแถบสีด้านข้าง (Colorbar)
        legend_kwds={
            'label': 'Number of A. baumannii Infections (2024)', 
            'orientation': 'vertical',
            'shrink': 0.6             # ย่อขนาดแถบสีให้พอดีกับรูป
        },
        missing_kwds={'color': 'lightgrey'}, # ถ้าเขตไหนไม่มีข้อมูลให้เป็นสีเทา
        ax=ax
    )

    # ตกแต่งกราฟ
    plt.title('Acinetobacter baumannii Infections by Health Region (2024)', fontsize=15, pad=15)
    ax.axis('off') # ปิดกรอบแกน X, Y
    plt.tight_layout()

    # ==========================================
    # 5. บันทึกรูปภาพ
    # ==========================================
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ สร้างแผนที่เสร็จสมบูรณ์! บันทึกไฟล์ไว้ที่:\n{SAVE_PATH}")

if __name__ == "__main__":
    plot_health_region_map()