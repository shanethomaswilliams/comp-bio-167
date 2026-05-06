import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- GLOBAL DATA ---
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

best_baseline_score = 0.52922  # ProtGoat
mean_baseline_score = 0.48963  # Mean of the baselines

def get_simulated_rank(model_score, leaderboard_scores):
    return (leaderboard_scores > model_score).sum() + 1

def create_f1max_graphic(model_data, prefix):
    print(f"Generating {prefix} F1Max improvement graphic...")
    
    plot_data = []
    for item in model_data:
        plot_data.append({
            "Model": item["Model"],
            "Comparison": "vs. Best Baseline (ProtGoat)",
            "Improvement": item["Score"] - best_baseline_score
        })
        plot_data.append({
            "Model": item["Model"],
            "Comparison": "vs. Mean of Baselines",
            "Improvement": item["Score"] - mean_baseline_score
        })
        
    df = pd.DataFrame(plot_data)
    
    sns.set_theme(style="whitegrid", context="talk")
    
    colors = {
        "vs. Best Baseline (ProtGoat)": "#8bd3e6", 
        "vs. Mean of Baselines": "#0078ff"         
    }
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=800)
    
    sns.barplot(
        data=df, y="Model", x="Improvement", hue="Comparison",
        palette=colors, edgecolor="white", linewidth=2, ax=ax
    )
    
    ax.axvline(x=0, color='#303030', linewidth=2.5, linestyle='-', zorder=3)
    ax.set_title(f"Model Improvement vs. Baselines (Δ F1Max) - {prefix}", fontweight='bold', fontsize=22, pad=20)
    ax.set_xlabel("F1Max Score Improvement", fontweight='bold', fontsize=16, labelpad=15)
    ax.set_ylabel("", fontweight='bold') 
    plt.yticks(fontweight='bold', fontsize=13)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%+.4f', padding=8, fontsize=10, fontweight='bold', color='#606060')
        
    leg = ax.legend(title="Baseline Comparison", loc='lower right', frameon=True)
    plt.setp(leg.get_title(), fontweight='bold')
    
    max_val = df["Improvement"].max()
    min_val = df["Improvement"].min()
    padding = (max_val - min_val) * 0.15
    ax.set_xlim(min_val - padding, max_val + padding)

    plt.tight_layout()
    filename = f"{prefix}_F1Max_Improvement_Graphic.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def create_rank_improvement_graphic(model_data, prefix):
    print(f"Generating {prefix} Rank improvement graphic...")
    csv_path = "/Users/shanewilliams/Desktop/TEMP_FOR_SHIT/cafa-5-protein-function-prediction-privateleaderboard-2026-04-22T05:18:27.csv" 
    
    try:
        lb_df = pd.read_csv(csv_path)
        leaderboard_scores = lb_df['Score'].astype(float)
    except FileNotFoundError:
        leaderboard_scores = pd.Series([
            0.61623, 0.58240, 0.57276, 0.56245, 0.56171, 0.56076, 0.55752, 0.55539, 0.55399, 
            0.54949, 0.54932, 0.54926, 0.54839, 0.54835, 0.54680, 0.54657, 0.54565, 
            0.540, 0.535, 0.530, 0.525, 0.520, 0.510, 0.500, 0.490, 0.480, 0.470, 0.450, 0.420, 0.410
        ])

    best_baseline_rank = get_simulated_rank(best_baseline_score, leaderboard_scores)
    mean_baseline_rank = get_simulated_rank(mean_baseline_score, leaderboard_scores)
    
    plot_data = []
    for item in model_data:
        model_rank = get_simulated_rank(item["Score"], leaderboard_scores)
        label_name = f"{item['Model']}\n[Final Rank: {model_rank}]"
        
        plot_data.append({
            "Model": label_name,
            "Comparison": "vs. Best Baseline (ProtGoat)",
            "Rank Improvement": best_baseline_rank - model_rank
        })
        plot_data.append({
            "Model": label_name,
            "Comparison": "vs. Mean of Baselines",
            "Rank Improvement": mean_baseline_rank - model_rank
        })
        
    df = pd.DataFrame(plot_data)

    sns.set_theme(style="whitegrid", context="talk")
    
    colors = {
        "vs. Best Baseline (ProtGoat)": "#ff0040", 
        "vs. Mean of Baselines": "#8b0000"         
    }
    
    fig, ax = plt.subplots(figsize=(14, 11), dpi=300) 
    
    sns.barplot(
        data=df, y="Model", x="Rank Improvement", hue="Comparison",
        palette=colors, edgecolor="white", linewidth=2, ax=ax
    )
    
    ax.axvline(x=0, color='#303030', linewidth=2.5, linestyle='-', zorder=3)
    ax.set_title(f"Leaderboard Rank Improvement vs. Baselines - {prefix}", fontweight='bold', fontsize=22, pad=20)
    ax.set_xlabel("Positions Gained / Lost on Leaderboard", fontweight='bold', fontsize=16, labelpad=15)
    ax.set_ylabel("", fontweight='bold') 
    
    plt.yticks(fontweight='bold', fontsize=12)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%+g', padding=8, fontsize=10, fontweight='bold', color='#606060')
        
    leg = ax.legend(title="Baseline Comparison", loc='lower right', frameon=True)
    plt.setp(leg.get_title(), fontweight='bold')
    
    max_val = df["Rank Improvement"].max()
    min_val = df["Rank Improvement"].min()
    padding = (max_val - min_val) * 0.15
    ax.set_xlim(min_val - padding, max_val + padding)

    plt.tight_layout()
    filename = f"{prefix}_Rank_Improvement_Graphic.png"
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def format_and_sort_data(raw_data):
    """Formats the label to include the tier, and sorts by score descending."""
    formatted = []
    for d in raw_data:
        formatted.append({
            "Model": f"{d['Model']} ({d['Group']})",
            "Score": d["Score"]
        })
    return sorted(formatted, key=lambda x: x["Score"], reverse=True)

if __name__ == "__main__":
    datasets = [
        ("Vanilla", vanilla_data),
        ("GoDAG", godag_data)
    ]
    
    for prefix, raw_data in datasets:
        formatted_model_data = format_and_sort_data(raw_data)
        
        # Run generating functions
        create_f1max_graphic(formatted_model_data, prefix)
        create_rank_improvement_graphic(formatted_model_data, prefix)
        print("-" * 40)
        
    print("\nSuccess! All graphics have been generated.")