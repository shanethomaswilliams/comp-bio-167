import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def create_updated_third_width_bar():
    print("Generating updated 1/3rd width aspect columns with new colors...")
    
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Base X positions for the groups
    w_full = 0.8
    w_third = w_full / 3

    # NEW COLORS: Spectrum of blue for aspects, dark blue for mean, red for test
    c_mfo = '#99ccff'   # Light blue
    c_bpo = '#66a3ff'   # Medium blue
    c_cco = '#3377ff'   # Standard blue
    c_mean = '#001a4d'  # Much darker blue
    c_test = '#ff3333'  # Bright Red
    
    # --- BASIC MEAN ENSEMBLE ---
    ax.bar(0 - w_third, 0.6438, width=w_third, color=c_mfo, edgecolor='w', linewidth=1)
    ax.bar(0, 0.5106, width=w_third, color=c_bpo, edgecolor='w', linewidth=1)
    ax.bar(0 + w_third, 0.6750, width=w_third, color=c_cco, edgecolor='w', linewidth=1)
    
    ax.bar(1, 0.6098, width=w_full, color=c_mean, edgecolor='w', linewidth=2)
    ax.bar(2, 0.5549, width=w_full, color=c_test, edgecolor='w', linewidth=2)
    
    # --- MOST COMPLEX ENSEMBLE ---
    ax.bar(4 - w_third, 0.7830, width=w_third, color=c_mfo, edgecolor='w', linewidth=1)
    ax.bar(4, 0.5857, width=w_third, color=c_bpo, edgecolor='w', linewidth=1)
    ax.bar(4 + w_third, 0.8124, width=w_third, color=c_cco, edgecolor='w', linewidth=1)
    
    ax.bar(5, 0.7270, width=w_full, color=c_mean, edgecolor='w', linewidth=2)
    ax.bar(6, 0.5292, width=w_full, color=c_test, edgecolor='w', linewidth=2)

    # Adding Labels inside the 1/3rd columns (vertical so they fit)
    ax.text(0 - w_third, 0.6438/2, "MFO", rotation=90, ha='center', va='center', color='black', weight='bold', size=10)
    ax.text(0, 0.5106/2, "BPO", rotation=90, ha='center', va='center', color='black', weight='bold', size=10)
    ax.text(0 + w_third, 0.6750/2, "CCO", rotation=90, ha='center', va='center', color='black', weight='bold', size=10)

    ax.text(4 - w_third, 0.7830/2, "MFO", rotation=90, ha='center', va='center', color='black', weight='bold', size=10)
    ax.text(4, 0.5857/2, "BPO", rotation=90, ha='center', va='center', color='black', weight='bold', size=10)
    ax.text(4 + w_third, 0.8124/2, "CCO", rotation=90, ha='center', va='center', color='black', weight='bold', size=10)

    # Top value labels
    # Basic
    ax.text(0 - w_third, 0.6438 + 0.01, ".644", ha='center', va='bottom', weight='bold', size=10, rotation=90)
    ax.text(0, 0.5106 + 0.01, ".511", ha='center', va='bottom', weight='bold', size=10, rotation=90)
    ax.text(0 + w_third, 0.6750 + 0.01, ".675", ha='center', va='bottom', weight='bold', size=10, rotation=90)
    ax.text(1, 0.6098 + 0.01, "0.6098", ha='center', va='bottom', weight='bold', size=13)
    ax.text(2, 0.5549 + 0.01, "0.5549", ha='center', va='bottom', weight='bold', size=13)
    
    # Complex
    ax.text(4 - w_third, 0.7830 + 0.01, ".783", ha='center', va='bottom', weight='bold', size=10, rotation=90)
    ax.text(4, 0.5857 + 0.01, ".586", ha='center', va='bottom', weight='bold', size=10, rotation=90)
    ax.text(4 + w_third, 0.8124 + 0.01, ".812", ha='center', va='bottom', weight='bold', size=10, rotation=90)
    ax.text(5, 0.7270 + 0.01, "0.7270", ha='center', va='bottom', weight='bold', size=13)
    ax.text(6, 0.5292 + 0.01, "0.5292", ha='center', va='bottom', weight='bold', size=13)

    # Group Labels at the bottom
    ax.set_xticks([1, 5])
    ax.set_xticklabels(["Mean", "Soft MoE"], weight='bold', size=16)

    # Titles and Axes
    ax.set_title("Aspect Splits vs CV Mean vs Test", fontweight='bold', fontsize=22, pad=20)
    ax.set_ylabel("F1Max Weighted Score", fontweight='bold', fontsize=16)
    
    ax.xaxis.grid(False) 
    ax.yaxis.grid(True, color='#f0f0f0', linestyle='-', linewidth=1)
    ax.set_axisbelow(True) 
    
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#303030')
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('#303030')
    ax.spines['left'].set_linewidth(1.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim(0, 0.95)

    # Legend
    legend_elements = [
        Patch(facecolor=c_mfo, edgecolor='w', label='MFO (Aspect)'),
        Patch(facecolor=c_bpo, edgecolor='w', label='BPO (Aspect)'),
        Patch(facecolor=c_cco, edgecolor='w', label='CCO (Aspect)'),
        Patch(facecolor=c_mean, edgecolor='w', label='CV Mean'),
        Patch(facecolor=c_test, edgecolor='w', label='TEST Score')
    ]
    leg = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), title="Metrics")
    plt.setp(leg.get_title(), fontsize='15', fontweight='bold')

    plt.tight_layout()
    filename = "Updated_Colors_Third_Width.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

if __name__ == "__main__":
    create_updated_third_width_bar()