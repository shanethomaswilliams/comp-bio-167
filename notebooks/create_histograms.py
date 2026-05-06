import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_histograms():
    print("Loading train_terms.tsv...")
    terms_df = pd.read_csv('/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/final_data_v3/train_terms.tsv', sep='\t')
    true_labels_df = terms_df[['EntryID', 'term']].drop_duplicates().copy()
    true_labels_df['is_true'] = True

    print("Processing train_merged.tsv to build histogram distributions (ignoring 0.0)...")
    chunksize = 2_000_000
    
    hist_data = {
        'Sequence': {'Correct': [], 'Incorrect': []},
        'Structure': {'Correct': [], 'Incorrect': []},
        'ProtGOAT': {'Correct': [], 'Incorrect': []}
    }

    Incorrect_SAMPLE_RATE = 0.05  
    HIGHLIGHT_SAMPLE_RATE = 0.20   

    for chunk in pd.read_csv('/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/final_data_v3/train_merged.tsv', sep='\t', chunksize=chunksize):
        merged = chunk.merge(
            true_labels_df, left_on=['protein_id', 'GO_term'], right_on=['EntryID', 'term'], how='left'
        )
        
        true_mask = merged['is_true'] == True
        pos_chunk = merged[true_mask]
        neg_chunk = merged[~true_mask]

        # 1. Sequence (Filter > 0.0)
        if not pos_chunk.empty:
            pos_seq = pos_chunk[pos_chunk['conf_sequence'] > 0.0]['conf_sequence']
            if not pos_seq.empty: hist_data['Sequence']['Correct'].append(pos_seq.sample(frac=HIGHLIGHT_SAMPLE_RATE, random_state=42))
        if not neg_chunk.empty:
            neg_seq = neg_chunk[neg_chunk['conf_sequence'] > 0.0]['conf_sequence']
            if not neg_seq.empty: hist_data['Sequence']['Incorrect'].append(neg_seq.sample(frac=Incorrect_SAMPLE_RATE, random_state=42))

        # 2. Structure (Filter > 0.0)
        if not pos_chunk.empty:
            pos_struct = pos_chunk[pos_chunk['conf_structure'] > 0.0]['conf_structure']
            if not pos_struct.empty: hist_data['Structure']['Correct'].append(pos_struct.sample(frac=HIGHLIGHT_SAMPLE_RATE, random_state=42))
        if not neg_chunk.empty:
            neg_struct = neg_chunk[neg_chunk['conf_structure'] > 0.0]['conf_structure']
            if not neg_struct.empty: hist_data['Structure']['Incorrect'].append(neg_struct.sample(frac=Incorrect_SAMPLE_RATE, random_state=42))

        # 3. ProtGOAT (Filter > 0.0)
        if not pos_chunk.empty:
            pos_goat = pos_chunk[pos_chunk['conf_protgoat'] > 0.0]['conf_protgoat']
            if not pos_goat.empty: hist_data['ProtGOAT']['Correct'].append(pos_goat.sample(frac=HIGHLIGHT_SAMPLE_RATE, random_state=42))
        if not neg_chunk.empty:
            neg_goat = neg_chunk[neg_chunk['conf_protgoat'] > 0.0]['conf_protgoat']
            if not neg_goat.empty: hist_data['ProtGOAT']['Incorrect'].append(neg_goat.sample(frac=Incorrect_SAMPLE_RATE, random_state=42))

    print("Data collected. Generating plots...")
    
    # Global visual settings
    sns.set_theme(style="whitegrid", context="talk")
    colors = {'Incorrect': '#0078ff', 'Correct': '#ff0000'}
    methods = ['Sequence', 'Structure', 'ProtGOAT']

    # --- 1. THE 1x3 GRID PLOT ---
    print("Generating 1x3 Histogram Grid...")
    fig_grid, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=300)
    
    for i, method in enumerate(methods):
        df_pos = pd.DataFrame({'Confidence': pd.concat(hist_data[method]['Correct']), 'Label': 'Correct'})
        df_neg = pd.DataFrame({'Confidence': pd.concat(hist_data[method]['Incorrect']), 'Label': 'Incorrect'})
        df_combined = pd.concat([df_neg, df_pos])

        x_min = df_combined['Confidence'].min()
        

        print(df_combined.head())

        sns.histplot(
            data=df_combined, x='Confidence', hue='Label', palette=colors,
            bins=100, stat='density', common_norm=False, kde=True, alpha=0.4, linewidth=0, ax=axes[i]
        )
        
        axes[i].set_title(f"{method} Confidence\n(Excluding 0.0)", fontweight='bold', pad=15)
        axes[i].set_xlabel(f"{method} Score", fontweight='bold')
        axes[i].set_xlim(x_min, 1.05)
        
        # Keep legend and Y-label only on the far left plot to keep the grid clean
        if i == 0:
            axes[i].set_ylabel("Density (Scaled Proportion)", fontweight='bold')
            leg = axes[i].get_legend()
            if leg:
                leg.set_title("Prediction Status")
                plt.setp(leg.get_title(), fontweight='bold')
        else:
            axes[i].set_ylabel("")
            if axes[i].get_legend():
                axes[i].get_legend().remove()

    plt.tight_layout()
    plt.savefig("Histograms_1x3_Grid_NoZeros.png", dpi=300, bbox_inches='tight')
    plt.close(fig_grid)
    print("Saved Histograms_1x3_Grid_NoZeros.png")


    # --- 2. THE INDIVIDUAL PLOTS ---
    print("Generating Individual Histograms...")
    for method in methods:
        df_pos = pd.DataFrame({'Confidence': pd.concat(hist_data[method]['Correct']), 'Label': 'Correct'})
        df_neg = pd.DataFrame({'Confidence': pd.concat(hist_data[method]['Incorrect']), 'Label': 'Incorrect'})
        df_combined = pd.concat([df_neg, df_pos])

        x_min = df_combined['Confidence'].min()

        fig_indiv, ax_indiv = plt.subplots(figsize=(10, 8), dpi=300)
        
        sns.histplot(
            data=df_combined, x='Confidence', hue='Label', palette=colors,
            bins=100, stat='density', common_norm=False, kde=True, alpha=0.4, linewidth=0, ax=ax_indiv
        )

        ax_indiv.set_title(f"{method} Confidence Distribution\n(Excluding 0.0 Predictions)", fontweight='bold', fontsize=22, pad=20)
        ax_indiv.set_xlabel(f"{method} Confidence Score", fontweight='bold', fontsize=16)
        ax_indiv.set_ylabel("Density (Scaled Proportion)", fontweight='bold', fontsize=16)
        ax_indiv.set_xlim(x_min, 1.05)

        # Enhance the legend for individual plots
        leg = ax_indiv.get_legend()
        if leg:
            leg.set_title("Prediction Status")
            plt.setp(leg.get_title(), fontsize='16', fontweight='bold')
            for lh in leg.legend_handles: 
                lh.set_alpha(1) # Make legend colors solid

        filename = f"Histogram_{method}_Indiv_NoZeros.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig_indiv)
        print(f"Saved {filename}")

    print("Done! All grid and individual histograms have been generated.")

if __name__ == "__main__":
    create_histograms()