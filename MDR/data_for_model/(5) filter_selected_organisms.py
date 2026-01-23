import pandas as pd
import os
import re

def filter_and_export_organisms():
    # 1. ตั้งค่าไฟล์
    input_file = os.path.join("MDR", "data_for_model", "AllYears_MDR_for_model.csv")
    output_combined = os.path.join("MDR", "data_for_model", "Selected_All_Pathogens.csv")
    
    print(f"📂 กำลังอ่านไฟล์: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: ไม่พบไฟล์ {input_file}")
        return

    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่ได้: {e}")
        return

    # ==========================================
    # 2. รายชื่อเชื้อที่ต้องการ (Target List)
    # ==========================================
    target_organisms = [
        "Escherichia coli",
        "Klebsiella pneumoniae",
        "Acinetobacter baumannii",
        "Pseudomonas aeruginosa",
        "Staphylococcus aureus",
        "Enterococcus faecalis",
        "Enterococcus faecium",
        "Streptococcus pneumoniae",
        "Haemophilus influenzae",
        "Salmonella spp."
    ]

    print(f"🎯 เป้าหมาย: คัดเลือกเชื้อ {len(target_organisms)} ชนิด")

    # Clean ชื่อในไฟล์ต้นฉบับก่อนเทียบ (เพื่อความชัวร์)
    if 'organism_full' in df.columns:
        df['organism_full'] = df['organism_full'].astype(str).str.strip()
    else:
        print("❌ ไม่พบคอลัมน์ 'organism_full'")
        return

    # ==========================================
    # 3. คัดแยกข้อมูล (Filtering)
    # ==========================================
    # เลือกเฉพาะแถวที่มีชื่อเชื้อตรงกับใน list
    df_selected = df[df['organism_full'].isin(target_organisms)].copy()
    
    found_count = len(df_selected)
    print(f"✅ พบข้อมูลที่ตรงกับรายชื่อทั้งหมด: {found_count:,} แถว")

    if found_count == 0:
        print("⚠️ ไม่พบข้อมูลเชื้อที่ระบุเลย (ลองเช็คตัวสะกดในไฟล์ต้นฉบับอีกครั้งครับ)")
        return

    # ==========================================
    # 4. บันทึกไฟล์รวม (Combined File)
    # ==========================================
    print(f"💾 กำลังบันทึกไฟล์รวม: {output_combined}")
    df_selected.to_csv(output_combined, index=False)

    # ==========================================
    # 5. บันทึกไฟล์แยกรายเชื้อ (Separate Files)
    # ==========================================
    print("-" * 50)
    print("📂 กำลังสร้างไฟล์แยกรายเชื้อ...")
    
    # วนลูปตามรายชื่อเชื้อที่ 'มีอยู่จริง' ในข้อมูลที่กรองมาแล้ว
    existing_organisms = df_selected['organism_full'].unique()

    for org_name in existing_organisms:
        # สร้างชื่อไฟล์ (แทนที่ช่องว่างด้วย _ เพื่อให้ชื่อไฟล์สวย)
        # เช่น Escherichia coli -> Escherichia_coli.csv
        safe_name = re.sub(r'[^\w\s-]', '', org_name).strip().replace(' ', '_')
        file_name = f"{safe_name}.csv"
        
        # เลือกข้อมูลเฉพาะเชื้อนั้น
        df_subset = df_selected[df_selected['organism_full'] == org_name]
        
        # บันทึก
        df_subset.to_csv(file_name, index=False)
        print(f"   -> สร้างไฟล์: {file_name} ({len(df_subset)} แถว)")

    # เช็คว่าตัวไหนหายไปไหม
    missing_orgs = set(target_organisms) - set(existing_organisms)
    if missing_orgs:
        print("-" * 50)
        print("⚠️ หมายเหตุ: ไม่พบเชื้อเหล่านี้ในไฟล์ข้อมูล:")
        for m in missing_orgs:
            print(f"   - {m}")
    
    print("-" * 50)
    print("🎉 เสร็จสมบูรณ์ครับ!")

if __name__ == "__main__":
    filter_and_export_organisms()