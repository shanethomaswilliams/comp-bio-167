import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_prediction_breakdown_chart():
    print("Processing train_merged.tsv in chunks to calculate exact prediction overlaps...")
    chunksize = 2_000_000
    
    # Initialize strict, mutually exclusive counters
    counters = {
        'All 3': 0,
        'Seq + Struct': 0,
        'Seq + ProtGOAT': 0,
        'Struct + ProtGOAT': 0,
        'Seq Only': 0,
        'Struct Only': 0,
        'ProtGOAT Only': 0
    }

    for chunk in pd.read_csv('/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/final_data_v3/train_merged.tsv', sep='\t', chunksize=chunksize):
        
        # Identify positive predictions
        has_seq = chunk['conf_sequence'] > 0.0
        has_struct = chunk['conf_structure'] > 0.0
        has_goat = chunk['conf_protgoat'] > 0.0
        
        # 1. All 3 Predicted
        counters['All 3'] += (has_seq & has_struct & has_goat).sum()
        
        # 2. Exactly 2 Predicted
        counters['Seq + Struct'] += (has_seq & has_struct & ~has_goat).sum()
        counters['Seq + ProtGOAT'] += (has_seq & ~has_struct & has_goat).sum()
        counters['Struct + ProtGOAT'] += (~has_seq & has_struct & has_goat).sum()
        
        # 3. Exactly 1 Predicted
        counters['Seq Only'] += (has_seq & ~has_struct & ~has_goat).sum()
        counters['Struct Only'] += (~has_seq & has_struct & ~has_goat).sum()
        counters['ProtGOAT Only'] += (~has_seq & ~has_struct & has_goat).sum()

    print("Counting complete! Generating bar chart...")

    # --- PLOTTING ---
    # Use 'white' style to clear out default heavy grids
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    bar_width = 0.6

    # X-Axis Positions
    pos_group1 = [1]
    pos_group2 = [3, 3.65, 4.30]
    pos_group3 = [6.5, 7.15, 7.80]

    # --- COLORS ---
    # Group 1: All 3
    color_all3 = '#3498db'         # Solid Blue
    
    # Group 2: Exactly 2 (Shades of Red: Lightest to Darkest)
    color_seq_struct = '#ff9999'   # Light Red
    color_seq_goat = '#e74c3c'     # Medium Red
    color_struct_goat = '#8b0000'  # Dark Red

    # Group 3: Only 1 (Shades of Grey: Lightest to Darkest)
    color_seq_only = '#d3d3d3'     # Light Grey
    color_struct_only = '#999999'  # Medium Grey
    color_goat_only = '#555555'    # Dark Grey

    # Plot Group 1 (All 3) - Blue
    bars_1 = ax.bar(pos_group1[0], counters['All 3'], width=bar_width, color=color_all3, edgecolor='black', label='All 3 Models')

    # Plot Group 2 (Exactly 2) - Reds
    bars_2a = ax.bar(pos_group2[0], counters['Seq + Struct'], width=bar_width, color=color_seq_struct, edgecolor='black', label='Seq + Struct')
    bars_2b = ax.bar(pos_group2[1], counters['Seq + ProtGOAT'], width=bar_width, color=color_seq_goat, edgecolor='black', label='Seq + ProtGOAT')
    bars_2c = ax.bar(pos_group2[2], counters['Struct + ProtGOAT'], width=bar_width, color=color_struct_goat, edgecolor='black', label='Struct + ProtGOAT')

    # Plot Group 3 (Exactly 1) - Greys
    bars_3a = ax.bar(pos_group3[0], counters['Seq Only'], width=bar_width, color=color_seq_only, edgecolor='black', label='Sequence Only')
    bars_3b = ax.bar(pos_group3[1], counters['Struct Only'], width=bar_width, color=color_struct_only, edgecolor='black', label='Structure Only')
    bars_3c = ax.bar(pos_group3[2], counters['ProtGOAT Only'], width=bar_width, color=color_goat_only, edgecolor='black', label='ProtGOAT Only')

    # --- FORMATTING & CLEAN BACKGROUND ---
    ax.set_title("Prediction Overlap Distribution", fontweight='bold', fontsize=22, pad=20)
    ax.set_ylabel("Number of Predictions", fontweight='bold', fontsize=16)
    
    # Custom X-Axis labels
    ax.set_xticks([1, 3.65, 7.15])
    ax.set_xticklabels(['All 3 Models\nPredicted', 'Exactly 2 Models\nPredicted', 'Only 1 Model\nPredicted'], fontweight='bold', fontsize=16)

    # Clean Grid: Only horizontal lines, made light and placed behind bars
    ax.grid(axis='y', color='#e0e0e0', linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True) # Ensure grid is behind the bars
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    # Function to add formatted numbers on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height):,}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5),  # 5 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)

    # Apply data labels
    add_labels([bars_1[0], bars_2a[0], bars_2b[0], bars_2c[0], bars_3a[0], bars_3b[0], bars_3c[0]])

    # Headroom for top labels
    max_height = max(counters.values())
    ax.set_ylim(0, max_height * 1.15)

    # Commas for Y-axis (e.g. 20,000,000)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    # Move legend outside the plot
    ax.legend(title="Prediction Combinations", title_fontsize=16, fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    
    filename = "Prediction_Overlap_Bar_Chart.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cleanly formatted chart as {filename}")

if __name__ == "__main__":
    generate_prediction_breakdown_chart()