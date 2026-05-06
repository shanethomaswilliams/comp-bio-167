import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- HARDCODED DATA ---
baselines = [
    {"Model": "Sequence", "Score": 0.48386, "Group": "Individual", "Complexity": 1, "Y_Offset": 0.001},
    {"Model": "Structure", "Score": 0.45581, "Group": "Individual", "Complexity": 1, "Y_Offset": -0.002},
    {"Model": "ProtGoat", "Score": 0.52922, "Group": "Individual", "Complexity": 1, "Y_Offset": 0.001},
]

vanilla_data = [
    {"Model": "Max", "Score": 0.55166, "Group": "Tier 1", "Complexity": 2, "Y_Offset": -0.002},
    {"Model": "Mean", "Score": 0.56812, "Group": "Tier 1", "Complexity": 2, "Y_Offset": 0.002},
    {"Model": "Rank Average", "Score": 0.55223, "Group": "Tier 1", "Complexity": 2, "Y_Offset": -0.005},
    {"Model": "Linear Regression", "Score": 0.54761, "Group": "Tier 2", "Complexity": 3, "Y_Offset": -0.002},
    {"Model": "Logistic Regression", "Score": 0.55013, "Group": "Tier 2", "Complexity": 3, "Y_Offset": 0.001},
    {"Model": "XGBoost", "Score": 0.54710, "Group": "Tier 2", "Complexity": 4, "Y_Offset": 0},
    {"Model": "MLP", "Score": 0.54657, "Group": "Tier 2", "Complexity": 5, "Y_Offset": 0.003},
    {"Model": "MLP + Emb", "Score": 0.53746, "Group": "Tier 3", "Complexity": 6, "Y_Offset": -0.003},
    {"Model": "Soft MoE", "Score": 0.54213, "Group": "Tier 3", "Complexity": 6, "Y_Offset": 0},
    {"Model": "Hard MoE", "Score": 0.52922, "Group": "Tier 3", "Complexity": 6, "Y_Offset": 0},
]

godag_data = [
    {"Model": "Max", "Score": 0.56414, "Group": "Tier 1", "Complexity": 2, "Y_Offset": -0.002},
    {"Model": "Mean", "Score": 0.57200, "Group": "Tier 1", "Complexity": 2, "Y_Offset": 0.002},
    {"Model": "Rank Average", "Score": 0.55166, "Group": "Tier 1", "Complexity": 2, "Y_Offset": -0.005},
    {"Model": "Linear Regression", "Score": 0.55847, "Group": "Tier 2", "Complexity": 3, "Y_Offset": -0.002},
    {"Model": "Logistic Regression", "Score": 0.55785, "Group": "Tier 2", "Complexity": 3, "Y_Offset": 0.001},
    {"Model": "XGBoost", "Score": 0.55651, "Group": "Tier 2", "Complexity": 4, "Y_Offset": 0},
    {"Model": "MLP", "Score": 0.55684, "Group": "Tier 2", "Complexity": 5, "Y_Offset": 0.003},
    {"Model": "MLP + Emb", "Score": 0.55165, "Group": "Tier 3", "Complexity": 6, "Y_Offset": -0.003},
    {"Model": "Soft MoE", "Score": 0.55460, "Group": "Tier 3", "Complexity": 6, "Y_Offset": 0},
    {"Model": "Hard MoE", "Score": 0.55559, "Group": "Tier 3", "Complexity": 6, "Y_Offset": 0},
]

# Shared plotting settings
colors = {
    'Individual': '#e0e0e0',
    'Tier 1': '#ffca3a',     
    'Tier 2': '#0078ff',     
    'Tier 3': '#ff0000',     
    'Tier 4': '#9d4edd'      
}

def draw_baselines_and_styling(ax):
    """Helper function to draw the dashed lines and axis styles"""
    seq_score = 0.48386
    struct_score = 0.45581
    goat_score = 0.52922
    mean_of_indivs = 0.48963  

    ax.axhline(y=seq_score, color='#e0e0e0', linestyle='--', linewidth=2, zorder=1)
    ax.axhline(y=struct_score, color='#e0e0e0', linestyle='--', linewidth=2, zorder=1)
    ax.axhline(y=goat_score, color='#e0e0e0', linestyle='--', linewidth=2, zorder=1)
    ax.axhline(y=mean_of_indivs, color='#808080', linestyle='--', linewidth=2.5, zorder=2)

    ax.text(7.2, goat_score + 0.001, 'ProtGoat', color='#b0b0b0', fontsize=10, weight='bold', ha='right')
    ax.text(7.2, mean_of_indivs + 0.001, 'Mean of Baselines', color='#808080', fontsize=10, weight='bold', ha='right')
    ax.text(7.2, seq_score - 0.004, 'Sequence', color='#b0b0b0', fontsize=10, weight='bold', ha='right')
    ax.text(7.2, struct_score + 0.001, 'Structure', color='#b0b0b0', fontsize=10, weight='bold', ha='right')

    ax.set_xlabel("Relative Model Complexity", fontweight='bold', fontsize=16, labelpad=15)
    ax.set_ylabel("F1Max Score", fontweight='bold', fontsize=16)

    x_ticks = [1, 2, 3, 4, 5, 6]
    x_labels = [
        "Individual\nBaselines", "Simple\nHeuristics", "Linear\nModels", 
        "Tree\nEnsembles", "Basic\nNeural Nets", "Embeddings\n& Routing",
    ]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')

    ax.set_xlim(0.5, 7.4) 
    ax.set_ylim(0.44, 0.58) # Slightly increased Y-limit to fit max scores

    ax.xaxis.grid(False) 
    ax.yaxis.grid(True, color='#f0f0f0', linestyle='-', linewidth=1) 
    
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#303030')
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('#303030')
    ax.spines['left'].set_linewidth(1.5)

def create_individual_scatter(model_data, title, filename):
    print(f"Generating {filename}...")
    
    df = pd.DataFrame(baselines + model_data)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=800)

    sns.scatterplot(
        data=df, x="Complexity", y="Score", hue="Group", palette=colors,
        s=300, edgecolor="w", linewidth=1.5, alpha=0.9, zorder=5, ax=ax
    )

    for i in range(df.shape[0]):
        ax.text(
            df.Complexity[i] + 0.1, df.Score[i] + df.Y_Offset[i], df.Model[i], 
            horizontalalignment='left', size=12, color='black', weight='bold'  
        )

    draw_baselines_and_styling(ax)
    ax.set_title(title, fontweight='bold', fontsize=22, pad=20)

    leg = ax.get_legend()
    if leg:
        leg.set_title("Model Tier")
        plt.setp(leg.get_title(), fontsize='16', fontweight='bold')
        for lh in leg.legend_handles: 
            lh.set_alpha(1) 
            if hasattr(lh, 'set_sizes'): lh.set_sizes([150])

    plt.tight_layout()
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    plt.close(fig)

def create_combined_scatter():
    print("Generating Combined Vanilla vs GoDAG scatter plot...")
    
    # Tag data for the combined view
    df_base = pd.DataFrame(baselines)
    df_base["Method"] = "Baseline"
    
    df_vanilla = pd.DataFrame(vanilla_data)
    df_vanilla["Method"] = "Vanilla"
    
    df_godag = pd.DataFrame(godag_data)
    df_godag["Method"] = "GoDAG"

    df = pd.concat([df_base, df_vanilla, df_godag], ignore_index=True)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=800)

    # Plot Baselines (circles)
    sns.scatterplot(
        data=df[df["Method"] == "Baseline"], x="Complexity", y="Score", hue="Group", palette=colors,
        marker="o", s=300, edgecolor="w", linewidth=1.5, alpha=0.9, zorder=5, ax=ax, legend=False
    )
    
    # Plot Vanilla (circles)
    sns.scatterplot(
        data=df[df["Method"] == "Vanilla"], x="Complexity", y="Score", hue="Group", palette=colors,
        marker="o", s=250, edgecolor="w", linewidth=1.5, alpha=0.7, zorder=6, ax=ax, legend=False
    )

    # Plot GoDAG (Pluses 'P')
    sns.scatterplot(
        data=df[df["Method"] == "GoDAG"], x="Complexity", y="Score", hue="Group", palette=colors,
        marker="P", s=350, edgecolor="w", linewidth=1.5, alpha=0.9, zorder=7, ax=ax
    )

    # Text annotations - Add offsets for Vanilla vs GoDAG so they don't perfectly overlap
    for i in range(df.shape[0]):
        method = df.Method.iloc[i]
        label = df.Model.iloc[i]
        
        # Suffix the labels to tell them apart, or just offset them slightly
        if method == "GoDAG":
            label = f"{label} (GoDAG)"
            y_shift = df.Y_Offset.iloc[i] + 0.002
            color = 'black'
        elif method == "Vanilla":
            label = f"{label} (Vanilla)"
            y_shift = df.Y_Offset.iloc[i] - 0.002
            color = '#505050'
        else:
            y_shift = df.Y_Offset.iloc[i]
            color = 'black'

        ax.text(
            df.Complexity.iloc[i] + 0.12, df.Score.iloc[i] + y_shift, label, 
            horizontalalignment='left', size=10, color=color, weight='bold'  
        )

    draw_baselines_and_styling(ax)
    ax.set_title("CAFA5 F1Max Score vs. Model Complexity (Vanilla vs. GoDAG)", fontweight='bold', fontsize=22, pad=20)

    # Customize the legend to show shape meanings
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='Vanilla / Baseline'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor='gray', markersize=12, label='GoDAG')
    ]
    
    leg = ax.legend(handles=custom_lines, title="Method", loc="lower right", frameon=True)
    plt.setp(leg.get_title(), fontsize='16', fontweight='bold')

    plt.tight_layout()
    filename = "Combined_Scatter_Complexity.png"
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

if __name__ == "__main__":
    create_individual_scatter(vanilla_data, "CAFA5 F1Max Score vs. Model Complexity (Vanilla)", "Vanilla_Scatter_Complexity.png")
    create_individual_scatter(godag_data, "CAFA5 F1Max Score vs. Model Complexity (GoDAG)", "GoDAG_Scatter_Complexity.png")
    create_combined_scatter()
    print("\nSuccess! All graphics have been generated.")