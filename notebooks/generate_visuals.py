import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_visualizations():
    # --- 1. DATA LOADING AND AGGREGATION ---
    print("Loading train_terms.tsv...")
    terms_df = pd.read_csv('/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/final_data_v3/train_terms.tsv', sep='\t')
    true_labels_df = terms_df[['EntryID', 'term']].drop_duplicates().copy()
    true_labels_df['is_true'] = True

    print("Processing train_merged.tsv in chunks to filter zeros...")
    chunksize = 2_000_000
    
    pos_3d_list, neg_3d_list, all_2d_list = [], [], []

    for chunk in pd.read_csv('/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/final_data_v3/train_merged.tsv', sep='\t', chunksize=chunksize):
        merged = chunk.merge(
            true_labels_df, left_on=['protein_id', 'GO_term'], right_on=['EntryID', 'term'], how='left'
        )
        merged['is_true'] = merged['is_true'].fillna(False)

        # 3D FILTERING (All three must be > 0)
        mask_3d = (merged['conf_sequence'] > 0.0) & (merged['conf_structure'] > 0.0) & (merged['conf_protgoat'] > 0.0)
        chunk_3d = merged[mask_3d]
        pos_3d_list.append(chunk_3d[chunk_3d['is_true'] == True].sample(frac=0.5, random_state=42))
        neg_3d_list.append(chunk_3d[chunk_3d['is_true'] == False].sample(frac=0.1, random_state=42))

        # 2D FILTERING
        mask_2d = ((merged['conf_sequence'] > 0) & (merged['conf_structure'] > 0)) | \
                  ((merged['conf_sequence'] > 0) & (merged['conf_protgoat'] > 0)) | \
                  ((merged['conf_structure'] > 0) & (merged['conf_protgoat'] > 0))
        
        chunk_2d = merged.loc[mask_2d, ['conf_sequence', 'conf_structure', 'conf_protgoat', 'is_true']].copy()
        chunk_2d[['conf_sequence', 'conf_structure', 'conf_protgoat']] = chunk_2d[['conf_sequence', 'conf_structure', 'conf_protgoat']].astype('float32')
        all_2d_list.append(chunk_2d)

    # --- 2. CALCULATE DYNAMIC BOUNDARIES ---
    # Helper function to find min/max and snap them if they are within 0.05 of 0 or 1
    def get_snapped_bounds(series):
        min_val = series.min()
        max_val = series.max()
        if min_val <= 0.05: min_val = 0.0
        if max_val >= 0.95: max_val = 1.0
        return min_val, max_val

    print("Finalizing 3D sample boundaries...")
    df_3d_pos = pd.concat(pos_3d_list)
    df_3d_neg = pd.concat(neg_3d_list)

    df_3d_pos = df_3d_pos.sample(n=min(20000, len(df_3d_pos)), random_state=42)
    df_3d_neg = df_3d_neg.sample(n=min(100000, len(df_3d_neg)), random_state=42)

    # Calculate 3D Bounds
    df_3d_combined = pd.concat([df_3d_pos, df_3d_neg])
    min_x, max_x = get_snapped_bounds(df_3d_combined['conf_sequence'])
    min_y, max_y = get_snapped_bounds(df_3d_combined['conf_structure'])
    min_z, max_z = get_snapped_bounds(df_3d_combined['conf_protgoat'])

    # --- 3. GENERATE 3D PLOT ---
    print("Generating scaled 3D Plotly visualization...")
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=df_3d_neg['conf_sequence'], y=df_3d_neg['conf_structure'], z=df_3d_neg['conf_protgoat'],
        mode='markers', marker=dict(size=3, color='#0078ff', opacity=0.1, line=dict(width=0)),
        name='Incorrect GO-Terms (Blue)', hoverinfo='none'
    ))

    fig.add_trace(go.Scatter3d(
        x=df_3d_pos['conf_sequence'], y=df_3d_pos['conf_structure'], z=df_3d_pos['conf_protgoat'],
        mode='markers', marker=dict(size=2.5, color='#ff0000', opacity=0.2, line=dict(width=0)),
        name='Correct GO-Terms (Red)', hoverinfo='none'
    ))

    axis_line = dict(color='black', width=6)
    fig.add_trace(go.Scatter3d(x=[min_x, max_x], y=[min_y, min_y], z=[min_z, min_z], mode='lines', line=axis_line, showlegend=False, hoverinfo='none'))
    fig.add_trace(go.Scatter3d(x=[min_x, min_x], y=[min_y, max_y], z=[min_z, min_z], mode='lines', line=axis_line, showlegend=False, hoverinfo='none'))
    fig.add_trace(go.Scatter3d(x=[min_x, min_x], y=[min_y, min_y], z=[min_z, max_z], mode='lines', line=axis_line, showlegend=False, hoverinfo='none'))

    cone_kw = dict(sizemode="absolute", sizeref=max(0.02, (max_x-min_x)*0.06), showscale=False, colorscale=[[0, 'black'], [1, 'black']], showlegend=False, hoverinfo='none')
    fig.add_trace(go.Cone(x=[max_x], y=[min_y], z=[min_z], u=[max_x-min_x], v=[0], w=[0], **cone_kw))
    fig.add_trace(go.Cone(x=[min_x], y=[max_y], z=[min_z], u=[0], v=[max_y-min_y], w=[0], **cone_kw))
    fig.add_trace(go.Cone(x=[min_x], y=[min_y], z=[max_z], u=[0], v=[0], w=[max_z-min_z], **cone_kw))

    txt_kw = dict(mode='text', showlegend=False, hoverinfo='none', textfont=dict(size=14, color='black'))
    pad_3d_x, pad_3d_y, pad_3d_z = (max_x - min_x) * 0.08, (max_y - min_y) * 0.08, (max_z - min_z) * 0.08

    fig.add_trace(go.Scatter3d(x=[max_x + pad_3d_x], y=[min_y], z=[min_z], text=[f'<b>{max_x:.2f} (Seq)</b>'], textposition='middle right', **txt_kw))
    fig.add_trace(go.Scatter3d(x=[min_x], y=[max_y + pad_3d_y], z=[min_z], text=[f'<b>{max_y:.2f} (Struct)</b>'], textposition='top center', **txt_kw))
    fig.add_trace(go.Scatter3d(x=[min_x], y=[min_y], z=[max_z + pad_3d_z], text=[f'<b>{max_z:.2f} (ProtGOAT)</b>'], textposition='top center', **txt_kw))
    fig.add_trace(go.Scatter3d(
        x=[min_x], y=[min_y], z=[min_z], mode='markers+text', marker=dict(size=6, color='black'), 
        text=[f'<b>Min:\n({min_x:.2f}, {min_y:.2f}, {min_z:.2f})</b>'], textposition='bottom left', showlegend=False, hoverinfo='none'
    ))

    blank_axis = dict(showgrid=False, zeroline=False, showline=False, showbackground=False, showaxeslabels=False, showticklabels=False, title='')
    fig.update_layout(
        title=dict(text="Ensemble Confidence Scores", font=dict(size=20)),
        scene=dict(
            xaxis=dict(**blank_axis, range=[min_x - pad_3d_x*2, max_x + pad_3d_x*2]),
            yaxis=dict(**blank_axis, range=[min_y - pad_3d_y*2, max_y + pad_3d_y*2]),
            zaxis=dict(**blank_axis, range=[min_z - pad_3d_z*2, max_z + pad_3d_z*2]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(itemsizing='constant', font=dict(size=18), x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1),
        paper_bgcolor='white', plot_bgcolor='white'
    )
    fig.write_html("index.html", include_plotlyjs="cdn")
    print("Saved 3D_static_vectors_Filtered.html")


    # --- 4. GENERATE 2D PROJECTIONS ---
    print("Formatting full dataset for 2D Seaborn Projections...")
    df_2d_all = pd.concat(all_2d_list)
    df_2d_all['Label'] = np.where(df_2d_all['is_true'], 'Correct (Red)', 'Incorrect (Blue)')
    df_2d_all = df_2d_all.sort_values('is_true') 

    df_2d_all = df_2d_all.rename(columns={
        'conf_sequence': 'Sequence Confidence',
        'conf_structure': 'Structure Confidence',
        'conf_protgoat': 'ProtGOAT Confidence'
    })

    # Calculate exact 2D Bounds for the specific data
    bounds_2d = {
        'Sequence Confidence': get_snapped_bounds(df_2d_all['Sequence Confidence'][df_2d_all['Sequence Confidence'] > 0.0]),
        'Structure Confidence': get_snapped_bounds(df_2d_all['Structure Confidence'][df_2d_all['Structure Confidence'] > 0.0]),
        'ProtGOAT Confidence': get_snapped_bounds(df_2d_all['ProtGOAT Confidence'][df_2d_all['ProtGOAT Confidence'] > 0.0])
    }

    unique_pairs = [
        ('Sequence Confidence', 'Structure Confidence'),
        ('Sequence Confidence', 'ProtGOAT Confidence'),
        ('Structure Confidence', 'ProtGOAT Confidence')
    ]

    sns.set_theme(style="whitegrid", context="talk")
    palette = {'Incorrect (Blue)': '#0078ff', 'Correct (Red)': '#ff0000'}
    scatter_kwargs = dict(alpha=0.005, s=3, edgecolor=None, rasterized=True)

    print("Generating 2D 1x3 Grid...")
    fig_grid, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=300)
    for i, (x_col, y_col) in enumerate(unique_pairs):
        mask_2d_pair = (df_2d_all[x_col] > 0.0) & (df_2d_all[y_col] > 0.0)
        plot_data = df_2d_all[mask_2d_pair]
        
        sns.scatterplot(data=plot_data, x=x_col, y=y_col, hue='Label', palette=palette, ax=axes[i], **scatter_kwargs)
        
        axes[i].set_title(f"{y_col} vs {x_col}", fontweight='bold', pad=15)

        # Apply Dynamic Min/Max Bounds with slight 2% visual padding
        min_x_2d, max_x_2d = bounds_2d[x_col]
        min_y_2d, max_y_2d = bounds_2d[y_col]
        pad_x_2d = (max_x_2d - min_x_2d) * 0.02
        pad_y_2d = (max_y_2d - min_y_2d) * 0.02

        axes[i].set_xlim(min_x_2d - pad_x_2d, max_x_2d + pad_x_2d)
        axes[i].set_ylim(min_y_2d - pad_y_2d, max_y_2d + pad_y_2d)

        if i > 0: axes[i].get_legend().remove() 
        else:
             leg = axes[i].get_legend()
             for lh in leg.legend_handles: lh.set_alpha(1)

    plt.tight_layout()
    plt.savefig('2D_Density_1x3_Grid_Filtered.png', dpi=300, bbox_inches='tight')
    plt.close(fig_grid)
    print("Saved 2D_Density_1x3_Grid_Filtered.png")

    print("Generating 2D Individual Plots...")
    for x_col, y_col in unique_pairs:
        mask_2d_pair = (df_2d_all[x_col] > 0.0) & (df_2d_all[y_col] > 0.0)
        plot_data = df_2d_all[mask_2d_pair]

        fig_indiv, ax_indiv = plt.subplots(figsize=(10, 8), dpi=300)
        sns.scatterplot(data=plot_data, x=x_col, y=y_col, hue='Label', palette=palette, ax=ax_indiv, **scatter_kwargs)
        
        ax_indiv.set_title(f"{y_col} vs {x_col}", fontweight='bold', fontsize=22, pad=20)
        ax_indiv.set_xlabel(x_col, fontweight='bold', fontsize=16)
        ax_indiv.set_ylabel(y_col, fontweight='bold', fontsize=16)

        # Apply Dynamic Min/Max Bounds with slight 2% visual padding
        min_x_2d, max_x_2d = bounds_2d[x_col]
        min_y_2d, max_y_2d = bounds_2d[y_col]
        pad_x_2d = (max_x_2d - min_x_2d) * 0.02
        pad_y_2d = (max_y_2d - min_y_2d) * 0.02

        ax_indiv.set_xlim(min_x_2d - pad_x_2d, max_x_2d + pad_x_2d)
        ax_indiv.set_ylim(min_y_2d - pad_y_2d, max_y_2d + pad_y_2d)
        
        leg = ax_indiv.legend(markerscale=8, title="Prediction Status", fontsize=14, title_fontsize=16)
        for lh in leg.legend_handles: 
            lh.set_alpha(1) 

        filename = f"2D_{x_col.split()[0]}_vs_{y_col.split()[0]}_Filtered.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig_indiv)
        print(f"Saved {filename}")

if __name__ == "__main__":
    create_visualizations()