import pandas as pd
import os

def check_standard_final_v2():
    # ==========================================
    # 1. Setup File Paths
    # ==========================================
    input_file = os.path.join("MDR", "data", "AllYears_processed.csv")
    standard_filename = "Drug_class_for_MDR_new.csv"
    
    # Check paths
    if os.path.exists(standard_filename):
        standard_file = standard_filename
    else:
        standard_file = os.path.join("MDR", "data_some_isolate", standard_filename)

    output_path = os.path.join("MDR", "data_some_isolate")
    os.makedirs(output_path, exist_ok=True)
    
    # Main Output Files
    output_all = os.path.join(output_path, "AllYears_Standard_Checked_All.csv")
    output_selected = os.path.join(output_path, "AllYears_Standard_Checked_Selected.csv")
    
    # New Summary Files
    summary_all_file = os.path.join(output_path, "Summary_Standard_All.csv")
    summary_selected_file = os.path.join(output_path, "Summary_Standard_Selected.csv")

    print(f"🚀 Starting Final Standard Check V2 (With Summary)...")

    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return
    if not os.path.exists(standard_file):
        print(f"❌ Error: Standard file not found: {standard_file}")
        return

    try:
        df_patient = pd.read_csv(input_file, low_memory=False)
        df_std = pd.read_csv(standard_file)
    except Exception as e:
        print(f"❌ Read File Error: {e}")
        return

    # ==========================================
    # 2. Preprocessing & Clean Headers
    # ==========================================
    df_patient.columns = df_patient.columns.str.strip().str.lower()
    df_std.columns = df_std.columns.str.strip().str.lower()

    # Standard columns config
    std_org_col = 'organism_who'  
    std_class_col = 'class'       
    std_drug_col = 'antibiotic'   

    # Check required columns
    if not all(col in df_std.columns for col in [std_org_col, std_class_col, std_drug_col]):
        print(f"❌ Error: Standard file missing columns.")
        return

    # Clean standard data
    df_std[std_org_col] = df_std[std_org_col].astype(str).str.strip().str.lower()
    df_std[std_class_col] = df_std[std_class_col].astype(str).str.strip()
    df_std[std_drug_col] = df_std[std_drug_col].astype(str).str.strip().str.lower()

    # Find organism column in patient file
    pt_org_col = 'organism_full'
    if pt_org_col not in df_patient.columns:
        pt_org_col = next((c for c in df_patient.columns if c in ['organism', 'bacteria']), None)
    
    if not pt_org_col:
        print("❌ Error: Organism column not found in patient data.")
        return

    # ==========================================
    # 3. Build Standard Dictionary
    # ==========================================
    print("📚 Building Standards Map...")
    standards_map = {}
    target_organisms = df_std[std_org_col].unique()
    
    for org in target_organisms:
        standards_map[org] = {}
        org_subset = df_std[df_std[std_org_col] == org]
        unique_classes = org_subset[std_class_col].unique()
        
        for cls in unique_classes:
            drugs = org_subset[org_subset[std_class_col] == cls][std_drug_col].tolist()
            valid_drugs = [d for d in drugs if d in df_patient.columns]
            
            if valid_drugs:
                standards_map[org][cls] = valid_drugs

    # ==========================================
    # 4. Check Compliance
    # ==========================================
    print("🔍 Checking row compliance...")
    valid_values = {'R', 'S', 'I'}

    def check_row_compliance(row):
        org_name = str(row[pt_org_col]).strip().lower()
        
        if org_name not in standards_map:
            return 0, 0, "", "", "", "No Standard"
        
        required_map = standards_map[org_name]
        tested_list = []
        missing_list = []
        
        for cls, drugs in required_map.items():
            is_tested = False
            for d in drugs:
                if str(row[d]).strip().upper() in valid_values:
                    is_tested = True
                    break
            
            if is_tested:
                tested_list.append(cls)
            else:
                missing_list.append(cls)
        
        total_req = len(required_map)
        count_tested = len(tested_list)
        
        str_tested = ", ".join(sorted(tested_list))
        str_missing = ", ".join(sorted(missing_list))
        str_required = ", ".join(sorted(required_map.keys()))
        
        if count_tested == total_req and total_req > 0:
            status = "Complete"
        elif total_req == 0:
             status = "No Standard (Empty)"
        else:
            status = f"Missing {len(missing_list)}"
        
        return total_req, count_tested, str_tested, str_missing, str_required, status

    result = df_patient.apply(check_row_compliance, axis=1, result_type='expand')
    
    df_patient['std_total_required'] = result[0]
    df_patient['std_tested_count'] = result[1]
    df_patient['std_tested_list'] = result[2]
    df_patient['std_missing_list'] = result[3]
    df_patient['std_required_list'] = result[4]
    df_patient['std_status'] = result[5]

    # ==========================================
    # 5. Save Main Files
    # ==========================================
    print(f"💾 Saving FILE 1 (All Rows): {output_all}")
    df_patient.to_csv(output_all, index=False)

    print(f"📂 Filtering for FILE 2 (Selected)...")
    mask_selected = df_patient[pt_org_col].astype(str).str.strip().str.lower().isin(standards_map.keys())
    df_selected = df_patient[mask_selected].copy()
    
    print(f"💾 Saving FILE 2 (Selected Rows): {output_selected}")
    df_selected.to_csv(output_selected, index=False)

    # ==========================================
    # 6. Generate Summaries (Group By & Count)
    # ==========================================
    print("-" * 60)
    print("📊 Generating Summaries (Group By Pattern)...")

    # Function to summarize a dataframe
    def generate_summary(df, filename):
        if df.empty:
            print(f"⚠️ Warning: DataFrame for {filename} is empty.")
            return

        # Group by the key columns to find duplicates
        summary = df.groupby(
            [pt_org_col, 'std_status', 'std_tested_list', 'std_missing_list']
        ).size().reset_index(name='count') # 'count' column stores the number of duplicates

        # Sort for better readability: Organism -> Status -> Count (High to Low)
        summary = summary.sort_values(
            by=[pt_org_col, 'std_status', 'count'], 
            ascending=[True, True, False]
        )

        print(f"💾 Saving SUMMARY: {filename}")
        summary.to_csv(filename, index=False)
        return summary

    # Generate Summary for ALL
    summary_all = generate_summary(df_patient, summary_all_file)

    # Generate Summary for SELECTED
    summary_selected = generate_summary(df_selected, summary_selected_file)

    print("-" * 60)
    print("✅ Process Completed Successfully!")
    print("Files Created:")
    print(f"1. Data (All):      {os.path.basename(output_all)}")
    print(f"2. Data (Selected): {os.path.basename(output_selected)}")
    print(f"3. Summary (All):   {os.path.basename(summary_all_file)}")
    print(f"4. Summary (Sel.):  {os.path.basename(summary_selected_file)}")
    print("-" * 60)
    
    if summary_selected is not None and not summary_selected.empty:
        print("Example Summary (Top 5 rows from Selected):")
        print(summary_selected.head(5).to_string(index=False))

if __name__ == "__main__":
    check_standard_final_v2()