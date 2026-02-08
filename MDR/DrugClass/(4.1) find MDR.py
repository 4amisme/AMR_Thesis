import pandas as pd
import os
import numpy as np

def check_standard_final_v9():
    # ==========================================
    # 1. ตั้งค่าไฟล์
    # ==========================================
    input_file = os.path.join("MDR", "data", "AllYears_processed.csv")
    standard_filename = "Drug_class_for_MDR_new.csv"
    
    if os.path.exists(standard_filename):
        standard_file = standard_filename
    else:
        standard_file = os.path.join("MDR", "data_some_isolate", standard_filename)

    output_path = os.path.join("MDR", "data_some_isolate")
    os.makedirs(output_path, exist_ok=True)
    
    # ไฟล์หลักชื่อเดิม (ยาวหน่อย)
    output_selected = os.path.join(output_path, "AllYears_Standard_Checked_Selected.csv")
    
    # ✅ ไฟล์ชื่อใหม่ (สั้นลง)
    output_short_file = os.path.join(output_path, "Selected_MDR.csv")

    print(f"🚀 Starting V9: Generating Detailed Data (Short Filename)...")

    # Check Files
    if not os.path.exists(input_file) or not os.path.exists(standard_file):
        print(f"❌ Error: ไม่พบไฟล์ Input หรือ Standard")
        return

    try:
        df_patient = pd.read_csv(input_file, low_memory=False)
        df_std = pd.read_csv(standard_file)
    except Exception as e:
        print(f"❌ Read File Error: {e}")
        return

    # ==========================================
    # 2. Preprocessing
    # ==========================================
    df_patient.columns = df_patient.columns.str.strip().str.lower()
    df_std.columns = df_std.columns.str.strip().str.lower()

    # Find Year
    year_col = None
    for col in df_patient.columns:
        if col in ['x_year', 'year', 'collection_year', 'date_year']:
            year_col = col
            break
            
    if not year_col:
        print("⚠️ Warning: No Year column found -> Creating dummy 'x_year' = 2023")
        df_patient['x_year'] = 2023
        year_col = 'x_year'
    
    # Prepare Standards
    std_org_col = 'organism_who'
    std_class_col = 'class'
    std_drug_col = 'antibiotic'
    
    df_std[std_org_col] = df_std[std_org_col].astype(str).str.strip().str.lower()
    df_std[std_class_col] = df_std[std_class_col].astype(str).str.strip()
    df_std[std_drug_col] = df_std[std_drug_col].astype(str).str.strip().str.lower()

    pt_org_col = 'organism_full'
    if pt_org_col not in df_patient.columns:
        pt_org_col = next((c for c in df_patient.columns if c in ['organism', 'bacteria']), None)
        
    if not pt_org_col:
        print("❌ Error: Organism column not found.")
        return

    # Build Map
    standards_map = {}
    target_organisms = df_std[std_org_col].unique()
    for org in target_organisms:
        standards_map[org] = {}
        subset = df_std[df_std[std_org_col] == org]
        for cls in subset[std_class_col].unique():
            drugs = subset[subset[std_class_col] == cls][std_drug_col].tolist()
            valid = [d for d in drugs if d in df_patient.columns]
            if valid:
                standards_map[org][cls] = valid

    # ==========================================
    # 3. Logic Check (Row by Row)
    # ==========================================
    print("🔍 Processing Rows (Tested & Resistant Lists)...")
    
    def check_row(row):
        org_name = str(row[pt_org_col]).strip().lower()
        if org_name not in standards_map:
            return 0, 0, "", "", "", "No Standard", ""
        
        required_map = standards_map[org_name]
        tested_list = []
        missing_list = []
        resistant_list = []
        
        for cls, drugs in required_map.items():
            is_tested = False
            is_resistant = False
            for d in drugs:
                val = str(row[d]).strip().upper()
                if val in ['R', 'S', 'I']: is_tested = True
                if val == 'R': is_resistant = True
            
            if is_tested:
                tested_list.append(cls)
                if is_resistant: resistant_list.append(cls)
            else:
                missing_list.append(cls)
        
        total = len(required_map)
        count = len(tested_list)
        status = "Complete" if (count == total and total > 0) else (f"Missing {len(missing_list)}" if total > 0 else "No Standard (Empty)")
        
        return (total, count, 
                ", ".join(sorted(tested_list)), 
                ", ".join(sorted(missing_list)), 
                ", ".join(sorted(required_map.keys())), 
                status, 
                ", ".join(sorted(resistant_list)))

    result = df_patient.apply(check_row, axis=1, result_type='expand')
    
    df_patient['std_total_required'] = result[0]
    df_patient['std_tested_count'] = result[1]
    df_patient['std_tested_list'] = result[2]
    df_patient['std_missing_list'] = result[3]
    df_patient['std_required_list'] = result[4]
    df_patient['std_status'] = result[5]
    df_patient['std_resistant_list'] = result[6]

    # ==========================================
    # 4. Filter & Save
    # ==========================================
    print("📂 Filtering Selected Organisms...")
    mask_selected = df_patient[pt_org_col].astype(str).str.strip().str.lower().isin(standards_map.keys())
    df_selected = df_patient[mask_selected].copy()

    # Save File 1 (Original Name)
    print(f"💾 Saving Detailed File: {output_selected}")
    df_selected.to_csv(output_selected, index=False)

    # Save File 2 (New Short Name)
    print(f"💾 Saving Short Name File: {output_short_file}")
    df_selected.to_csv(output_short_file, index=False)

    print("-" * 60)
    print("✅ เสร็จสมบูรณ์! ได้ไฟล์ 2 ไฟล์ (เนื้อหาเหมือนกัน):")
    print(f"1. {os.path.basename(output_selected)} (ชื่อเดิม)")
    print(f"2. {os.path.basename(output_short_file)} (ชื่อใหม่ สั้นกระชับ)")
    print("-" * 60)

if __name__ == "__main__":
    check_standard_final_v9()