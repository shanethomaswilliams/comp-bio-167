import pandas as pd

# Set your file paths here
HR_PATH = "/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/results/ensemble_20260423_023530/hard_routing_test_selections.tsv"
SOFT_PATH = "/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/results/ensemble_20260423_032826/soft_moe_test_gates.tsv"

# ==========================================
# 1. HARD ROUTING ANALYSIS
# ==========================================
print("==========================================")
print(" 1. HARD ROUTING ANALYSIS ")
print("==========================================")
hr_df = pd.read_csv(HR_PATH, sep='\t')

# Calculate counts and percentages
hr_counts = hr_df['selected_model'].value_counts()
hr_pct = hr_df['selected_model'].value_counts(normalize=True) * 100

# Combine into a single readable dataframe
hr_summary = pd.DataFrame({
    'Count': hr_counts,
    'Percentage (%)': hr_pct.round(2)
})

print("Model Selection Breakdown:")
print(hr_summary)
print("\n")


# ==========================================
# 2. SOFT MOE ANALYSIS
# ==========================================
print("==========================================")
print(" 2. SOFT MOE ANALYSIS ")
print("==========================================")
soft_df = pd.read_csv(SOFT_PATH, sep='\t')
# print(soft_df.head())

# Filter only rows that actually had embeddings
if 'has_embedding' in soft_df.columns:
    soft_df = soft_df[soft_df['has_embedding'] == True]
else:
    print("Notice: 'has_embedding' column not found. Skipping filter.")

# Isolate just the base model columns
model_cols = [c for c in soft_df.columns if c not in ['protein_id', 'has_embedding']]
gate_weights = soft_df[model_cols]

# Calculate all requested statistics at once and transpose (.T) for readability
soft_stats = gate_weights.agg(['mean', 'median', 'min', 'max', 'std']).T

# Rename columns for a cleaner printout
soft_stats.columns = ['Mean', 'Median', 'Min', 'Max', 'Std Dev']

print("Gate Weight Distribution (Per Model):")
print(soft_stats.round(4))