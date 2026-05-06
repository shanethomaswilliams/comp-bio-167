import pandas as pd
import re
import os

COLS = ['protein_id', 'go_term', 'confidence']

def load_tsv(tsv_path):
    if tsv_path.endswith('.gz'):
        df = pd.read_csv(tsv_path, sep='\t', header=None, compression='gzip')
    else:
        df = pd.read_csv(tsv_path, sep='\t', header=None)
    df = df.iloc[:, :3]  # take only first 3 columns
    df.columns = COLS
    return df

def combine_datasets(datasets):
    return pd.concat(datasets, ignore_index=True)[COLS]

def save_tsv(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[COLS].to_csv(output_path, sep='\t', index=False, header=False)

def trim_go_terms(df, threshold_csv="..."):
    frequency_df = pd.read_csv(threshold_csv)
    frequency_set = set(frequency_df.iloc[:, 0])
    return df[df['go_term'].isin(frequency_set)]

def parse_protein_go_data(df):
    """Extract protein ID, GO term, and confidence score from annotation dataframe."""
    # Assuming the data is in a single column, split it appropriately
    df['protein_id'] = df.iloc[:, 0].str.split().str[0]
    df['go_term'] = df.iloc[:, 0].str.extract(r'(GO:\d+)')[0]
    df['confidence_score'] = df.iloc[:, 0].str.extract(r'(GO:\d+\s+([\d.]+))')[1]
    
    return df[['protein_id', 'go_term', 'confidence_score']]