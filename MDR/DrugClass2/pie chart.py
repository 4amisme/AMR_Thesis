import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Global Settings for Professional Look ---
# Set default font to a clean sans-serif font (e.g., Arial or Helvetica if available)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# Define a professional color palette (distinct, non-glaring colors)
# Deep Blue, Teal, Deep Red, Orange, Purple. Grey for 'Others'.
PROFESSIONAL_COLORS = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#A9A9A9']

def main():
    # --- 1. Setup Paths ---
    base_folder = os.path.join("MDR", "DrugClass2")
    input_file = os.path.join(base_folder, "AllYears_MDR_Pattern.csv")
    output_chart_folder = os.path.join(base_folder, "Presentation_Charts")

    if not os.path.exists(input_file):
        print(f"[Error] Input file not found: {input_file}")
        return

    if not os.path.exists(output_chart_folder):
        os.makedirs(output_chart_folder)
        print(f"Created output folder: {output_chart_folder}")

    # --- 2. Read Data ---
    print("Reading data...")
    df = pd.read_csv(input_file, encoding='utf-8')

    if 'organism_full' not in df.columns or 'Resistant_Drug_Classes' not in df.columns:
        print("[Error] Missing necessary columns in the data file.")
        return

    unique_organisms = df['organism_full'].unique()
    print(f"Found {len(unique_organisms)} organisms to process.")

    # --- 3. Loop and Create Charts ---
    for org in unique_organisms:
        print(f"\nProcessing organism: {org} ...")
        
        # Filter data for current organism
        subset = df[df['organism_full'] == org]
        total_cases = len(subset)
        
        if total_cases == 0:
            continue

        # Count patterns
        pattern_counts = subset['Resistant_Drug_Classes'].value_counts()
        
        # Select Top 5
        top5 = pattern_counts.head(5)
        
        # Calculate 'Others'
        top5_sum = top5.sum()
        others_count = total_cases - top5_sum
        
        # Prepare plotting data
        plot_labels = list(top5.index)
        plot_values = list(top5.values)
        
        # Assign colors based on the number of slices needed
        current_colors = PROFESSIONAL_COLORS[:len(plot_values)]

        if others_count > 0:
            plot_labels.append(f'Others ({len(pattern_counts)-5} other patterns)')
            plot_values.append(others_count)
            # Ensure grey is the last color for 'Others'
            if len(current_colors) < len(PROFESSIONAL_COLORS):
                 current_colors.append(PROFESSIONAL_COLORS[-1]) # Add grey

        # --- CREATE PROFESSIONAL DONUT CHART ---
        
        # Set figure size (larger for better resolution/spacing)
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Explosion parameters (highlight top 1)
        explode = [0.08] + [0]* (len(plot_values)-1) if len(plot_values) > 0 else None

        # Draw Pie
        wedges, texts, autotexts = ax.pie(
            plot_values,
            autopct='%1.1f%%',     # Show percentage
            startangle=140,        # A good starting angle for professional looks
            colors=current_colors,
            pctdistance=0.82,      # Push percentage text towards the edge
            explode=explode,
            shadow=True,           # Add subtle shadow for depth
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5} # Clean white separators
        )
        
        # Style the percentage text (white, bold, larger)
        plt.setp(autotexts, size=11, weight="bold", color="white")

        # Draw Center Circle (Donut)
        # Add text in the center showing Total N
        centre_circle = plt.Circle((0,0), 0.65, fc='white')
        fig.gca().add_artist(centre_circle)
        
        ax.text(0, 0, f"Total Cases (N)\n{total_cases:,}", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

        # --- Legend and Titles ---
        
        # Main Title
        plt.suptitle(f"Top 5 Multi-Drug Resistant (MDR) Patterns", 
                     fontsize=20, fontweight='bold', color='#222222', y=0.98)
        # Subtitle
        plt.title(f"Organism: {org}", fontsize=16, color='#555555', pad=20)

        # Legend formatting (crucial due to long names)
        ax.legend(
            wedges,
            plot_labels,
            title="MDR Resistance Profiles (A-Z Sorted)",
            title_fontsize=12,
            loc="center left",
            bbox_to_anchor=(1.05, 0, 0.5, 1), # Position outside right
            fontsize=11,
            frameon=False # Clean look without box border
        )

        plt.axis('equal') # Ensure it's a perfect circle
        plt.tight_layout()

        # Save high-resolution image for presentations
        safe_filename = str(org).replace(" ", "_").replace("/", "-") + "_ProfessionalChart.png"
        save_path = os.path.join(output_chart_folder, safe_filename)
        
        # dpi=300 is standard for high-quality print/presentation
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        plt.close()
        
        print(f"   ✅ Saved professional chart: {safe_filename}")

    print("\n--- Process Complete ---")
    print(f"All professional charts are saved in: {output_chart_folder}")

if __name__ == "__main__":
    main()