import pandas as pd
import sys
sys.path.append('../src')

from preprocessing import load_tsv, combine_datasets, parse_protein_go_data, save_tsv, trim_go_terms

test_df = load_tsv("/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/structure_team/supposed_final_predictions/test_predictions_combined.tsv")

# display train_df and its current length
print(test_df.head())
print(f"Length of train_df before trimming: {len(test_df)}")

test_df_10 = trim_go_terms(test_df, threshold_csv="/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/frequency_csvs/frequency_terms_10.csv")
test_df_50 = trim_go_terms(test_df, threshold_csv="/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/frequency_csvs/frequency_terms_50.csv")

print(test_df_10.head())
print(f"Length of train_df after trimming with threshold 10: {len(test_df_10)}")

print(test_df_50.head())
print(f"Length of train_df after trimming with threshold 50: {len(test_df_50)}")

save_tsv(test_df_10, "/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/structure_team/supposed_final_predictions/combined_my_trim10_test_predictions.tsv")
save_tsv(test_df_50, "/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/structure_team/supposed_final_predictions/combined_my_trim50_test_predictions.tsv")

final_test_df = pd.read_csv("/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/final_data/test_merged.tsv", sep='\t')
print(final_test_df.head())