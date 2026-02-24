import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

# 1. กำหนด Path
BASE_DIR = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/AMR_Trend/Spatiotemporal'
MAP_FILE = '/Users/chanokchonkarinrak/Documents/GitHub/AMR_Thesis/MDR/model_LR/Regions_no_province_boundaries.json'

def create_spatiotemporal_maps(target_drug):
    print(f"\n=====================================")
    print(f"🗺️ กำลังสร้างแผนที่สำหรับ: {target_drug}")
    
    # 2. โหลดไฟล์ผลลัพธ์ของ INLA ที่เพิ่งรันมา
    res_file = os.path.join(BASE_DIR, f"ST_ForecastResult_{target_drug}.csv")
    if not os.path.exists(res_file):
        print(f"ไม่พบไฟล์: {res_file}")
        return
        
    df = pd.read_csv(res_file)
    
    # 3. โหลดแผนที่ประเทศไทย (GeoJSON)
    if not os.path.exists(MAP_FILE):
        print(f"ไม่พบไฟล์แผนที่ GeoJSON: {MAP_FILE}")
        return
        
    thailand_map = gpd.read_file(MAP_FILE)
    # ตรวจสอบว่าคอลัมน์ชื่ออะไร (มักจะเป็น 'HealthRegion' หรือ 'REGION')
    map_region_col = 'HealthRegion' if 'HealthRegion' in thailand_map.columns else thailand_map.columns[0]
    thailand_map[map_region_col] = thailand_map[map_region_col].astype(float)
    
    # 4. คำนวณค่าเฉลี่ย Predicted %R รายปี (จากรายเดือน)
    # ใช้คอลัมน์ predicted_percent ที่ได้จาก INLA
    df_annual = df.groupby(['REGION', 'Year'])['predicted_percent'].mean().reset_index()
    
    # สร้างโฟลเดอร์สำหรับเซฟรูปแผนที่
    map_out_dir = os.path.join(BASE_DIR, 'Maps')
    os.makedirs(map_out_dir, exist_ok=True)
    
    # กำหนดช่วงสี (Color Scale) ให้อิงจากค่า Min-Max รวมทั้งหมด จะได้เปรียบเทียบสีข้ามปีได้
    vmin = df_annual['predicted_percent'].min()
    vmax = df_annual['predicted_percent'].max()
    cmap = 'OrRd' # สีโทน ขาว-ส้ม-แดง

    # 5. วาดแผนที่แยกทีละปี (ตั้งแต่ 2024 ถึง 2029)
    target_years = sorted(df_annual[df_annual['Year'] >= 2024]['Year'].unique())
    
    if len(target_years) == 0:
        print("ไม่พบข้อมูลปี 2024 ขึ้นไป ในไฟล์ผลลัพธ์")
        return

    # สร้างรูปตารางขนาดใหญ่รวมทุกปี (เช่น 2x3 ช่อง)
    cols = 3
    rows = (len(target_years) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*6))
    axes = axes.flatten()

    for idx, year in enumerate(target_years):
        ax = axes[idx]
        
        # ดึงข้อมูลเฉพาะปีนั้น
        yearly_data = df_annual[df_annual['Year'] == year]
        
        # นำข้อมูลมาเชื่อมกับแผนที่ (Join)
        merged_map = thailand_map.merge(yearly_data, left_on=map_region_col, right_on='REGION', how='left')
        
        # วาดแผนที่
        merged_map.plot(
            column='predicted_percent', 
            cmap=cmap, 
            linewidth=0.8, 
            ax=ax, 
            edgecolor='0.3', 
            vmin=vmin, 
            vmax=vmax,
            missing_kwds={'color': 'lightgrey'} 
        )
        
        ax.set_title(f"Forecast {year}", fontsize=14, fontweight='bold', pad=10)
        ax.axis('off') # ปิดกรอบแกน X, Y
        
        # ใส่ตัวเลข %R กำกับตรงกลางแต่ละเขต (Optional)
        merged_map['coords'] = merged_map['geometry'].representative_point()
        for _, row in merged_map.iterrows():
            if pd.notnull(row['predicted_percent']):
                ax.text(row['coords'].x, row['coords'].y, 
                        f"{row['predicted_percent']:.1f}%", 
                        horizontalalignment='center', fontsize=8, color='black',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # ปิดแกนในช่องที่ว่างเปล่า (ถ้ามี)
    for idx in range(len(target_years), len(axes)):
        axes[idx].axis('off')

    # ใส่แถบสี (Colorbar) รวมไว้ด้านล่าง
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Predicted Resistance Percentage (%R) - {target_drug}', fontsize=12, labelpad=10)

    plt.suptitle(f'Spatiotemporal Evolution of {target_drug} Resistance\n(Acinetobacter baumannii, 13 Health Regions)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    save_path = os.path.join(map_out_dir, f'Map_Evolution_{target_drug}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ บันทึกแผนที่รวมสำเร็จ: {save_path}")

if __name__ == "__main__":
    create_spatiotemporal_maps('Imipenem')
    create_spatiotemporal_maps('Meropenem')