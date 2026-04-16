import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import kagglehub

SEED = 42
PATH = "cafa-5-protein-function-prediction/"

"""
This file contain codes that will load T5 protein embeddings and taxomomy ids, 
and stacks all protein feature in to a nparray
The T5 embeddings are reduced dimensions with PCA, with 64 components for now

TODO:
load fasta data for each protein
"""

from transformers import T5Tokenizer, T5EncoderModel
import torch


"""
Extract ProtT5 embeddings for a dict of proteins.
This part of code loading embeddings is written by Claude
"""
def load_T5_model():
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
    model.eval()  # disable dropout — we are doing inference not training
    return tokenizer, model

def extract_T5_embeddings(sequences_dict, tokenizer, model, device='cpu'):

    model = model.to(device)
    embeddings = {}

    for protein_id, sequence in sequences_dict.items():
        # ProtT5 expects spaces between each amino acid character
        sequence_spaced = ' '.join(list(sequence))

        inputs = tokenizer(
            sequence_spaced,
            return_tensors='pt',
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0)
        embeddings[protein_id] = embedding.cpu().numpy()
    print(f"T5 dict loaded with {len(embeddings)} entries")
    return embeddings


def load_T5embd(sequences_dict):
    tokenizer, model = load_T5_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return extract_T5_embeddings(sequences_dict, tokenizer, model, device)


def load_taxon():
    terms_taxon = pd.read_csv(PATH + "train_taxonomy.tsv", sep='\t')
    # FIX: use dict(zip()) instead of manual loop
    taxon_dict = dict(zip(terms_taxon['EntryID'], terms_taxon['taxonomyID']))
    print(f"taxon dict loaded with {len(taxon_dict)} entries")
    return taxon_dict


def load_fasta(fasta_path):
    fasta_dict = {}
    current_id = None
    current_seq = []

    for i, filename in enumerate(fasta_path):
        with open(PATH + filename) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # save the previous protein before starting a new one
                    if current_id is not None:
                        seq = ''.join(current_seq)
                        fasta_dict[current_id] = {
                            'sequence' : seq,
                            'length'   : len(seq),
                        }
                    # the ID is the first word after '>'
                    current_id  = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

    if current_id is not None:
        seq = ''.join(current_seq)
        fasta_dict[current_id] = {
            'sequence' : seq,
            'length'   : len(seq),
        }
    print(f"fasta dict loaded with {len(fasta_dict)} entries")
    return fasta_dict


"""
Build a dictionary of each protein to its features
Each protein's value contains:
    - 'embedding', np.ndarray shape (emb_dim,)
    - 'taxon_id'
    - 'length'?
    - 'n_domains'?
"""
def build_protein_features(proteins, T5_dict, taxon_dict, fasta_dict):
    protein_features = {}
    for p in proteins:
        embedding = T5_dict[p]
        taxon_id  = taxon_dict[p]
        length    = fasta_dict[p]['length']
        # n_domains = fasta_dict[p]['n_domains']

        protein_features[p] = {
            'embedding' : embedding,
            'taxon_id'  : taxon_id,
            'length'    : length,
            # 'n_domains' : n_domains,
        }
    return protein_features


"""
Build feature matrix without PCA features.
To be changed if we want to include different features for protein.

taxon_to_idx is built once from training proteins in __main__ and passed in
here so that the same mapping is used consistently for both train and test,
and unseen taxa in test fall back to index 0.

arguments:
    merged_df        : DataFrame[protein_id, GO_term, conf_seq, conf_struct, conf_net]
    protein_features : dict {protein_id: feature_dict}
    taxon_to_idx     : dict {taxon_id: column_index} built from training proteins only
"""
def other_enriched_features(merged_df, protein_features, taxon_to_idx):
    conf_cols = ['conf_seq', 'conf_struct', 'conf_net']

    # convert the confidence features to a raw numpy array.
    X_conf = merged_df[conf_cols].values.astype(np.float32)  # (n_rows, 3)

    # one-hot taxon ID — look up each protein's taxon index using the
    # pre-built mapping; .get(..., 0) maps unseen taxa to column 0
    n_taxa = len(taxon_to_idx)
    taxon_ids = np.array([
        taxon_to_idx.get(protein_features[p]['taxon_id'], 0)
        for p in merged_df['protein_id']
    ])
    X_taxon = np.zeros((len(merged_df), n_taxa), dtype=np.float32)
    X_taxon[np.arange(len(merged_df)), taxon_ids] = 1.0

    # other protein features
    lengths   = np.array([
        protein_features[p]['length'] for p in merged_df['protein_id']
    ], dtype=np.float32)
    n_domains = np.array([
        protein_features[p]['n_domains'] for p in merged_df['protein_id']
    ], dtype=np.float32)

    # np.log1p(x) = log(1 + x) — a common feature-engineering trick for
    # count-like or length-like values: compresses the tail of long proteins
    # while keeping 0 mapped to 0 (log1p(0) = 0, whereas log(0) = -inf).
    log_length = np.log1p(lengths).reshape(-1, 1)   # (n_rows, 1)
    X_domains  = n_domains.reshape(-1, 1)            # (n_rows, 1)

    # Concatenate all parts horizontally
    X = np.hstack([X_conf, X_taxon, log_length, X_domains])  # (n_rows, total_dim)
    return X


"""
Building PCA and complete enriched feature matrix.
takes only train_protein_features (already filtered) instead of
the full dict
"""
def fit_pca_on_proteins(train_protein_features, n_components):
    train_embs = np.vstack([
        train_protein_features[p]['embedding']
        for p in train_protein_features
    ])

    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(train_embs)
    explained = pca.explained_variance_ratio_.sum()
    print(f'PCA ({n_components} components) explains {explained:.1%} of variance')
    return pca


def transform_pca(merged_df, protein_features, pca_model):
    # Compress each unique protein's embedding once, then broadcast to all rows
    unique_proteins = merged_df['protein_id'].unique()
    raw_embs = np.vstack([
        protein_features[p]['embedding'] for p in unique_proteins
    ])
    pca_embs = pca_model.transform(raw_embs).astype(np.float32)  # (n_unique, PCA_DIM)

    # embedding rows instead of indices. Correct form maps protein_id → embedding row.
    prot_to_pca = {p: pca_embs[i] for i, p in enumerate(unique_proteins)}

    # For each row in merged_df, retrieve that protein's PCA embedding
    X_emb = []
    for prot in merged_df['protein_id']:
        X_emb.append(prot_to_pca[prot])

    return np.vstack(X_emb)  # (n_rows, PCA_DIM)


def build_complete_feature_matrix(X, X_emb):
    X_with_pca = np.hstack([X, X_emb])
    return X_with_pca


def load_files(args):
    tr_dfs = []
    for i, filename in enumerate(args.tr):
        df = pd.read_csv(PATH + filename,
                         sep='\t',
                         header=None,
                         names=['protein_id', 'GO_term', 'conf_'+str(i)])
        df = df.set_index(['protein_id', 'GO_term'])
        tr_dfs.append(df)
    merged_tr_conf = pd.concat(tr_dfs, axis=1, join='outer').fillna(0.0).reset_index()
    print(f"train conf loaded with {len(merged_tr_conf)} entries")

    labels = merged_tr_conf[['protein_id', 'GO_term']].copy()
    train_terms = pd.read_csv(PATH + "Train/train_terms.tsv", sep='\t')
    true_pairs = set(zip(train_terms['EntryID'], train_terms['term']))
    labels['label'] = [
        int((row.protein_id, row.GO_term) in true_pairs)
        for row in labels.itertuples(index=False)
    ]
    print(f"train terms loaded with {len(train_terms)} entries")

    test_dfs = []
    for i, filename in enumerate(args.te):
        df = pd.read_csv(PATH + filename,
                         sep='\t',
                         header=None,
                         names=['protein_id', 'GO_term', 'conf_'+str(i)])
        df = df.set_index(['protein_id', 'GO_term'])
        test_dfs.append(df)
    merged_test_conf = pd.concat(test_dfs, axis=1, join='outer').fillna(0.0).reset_index()
    print(f"test conf loaded with {len(merged_test_conf)} entries")

    ia_df = pd.read_csv(f'{PATH}/IA.txt', sep='\t', header=None, names=['term', 'ia'])
    IA_WEIGHTS = dict(zip(ia_df['term'], ia_df['ia']))
    print(f"ia weights loaded with {len(IA_WEIGHTS)} entries")

    return merged_tr_conf, labels, merged_test_conf, IA_WEIGHTS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', nargs='+', required=True)
    parser.add_argument('-te', nargs='+', required=True)
    parser.add_argument('-fasta', nargs='+', required=True)
    args = parser.parse_args()
    merged_tr_conf, labels, merged_test_conf, IA_WEIGHTS = load_files(args)

    # Load protein data source
    fasta_dict = load_fasta(args.fasta)  # load sequences first
    T5_dict = load_T5embd(fasta_dict)  # compute embeddings from sequences
    taxon_dict = load_taxon()

    # build protein feature dicts
    tr_proteins = merged_tr_conf['protein_id'].unique()
    te_proteins = merged_test_conf['protein_id'].unique()

    tr_features = build_protein_features(tr_proteins, T5_dict, taxon_dict, fasta_dict)
    te_features = build_protein_features(te_proteins, T5_dict, taxon_dict, fasta_dict)

    # Build taxon_to_idx from training proteins only
    # ensure same mapping for both tr and test sets
    unique_taxa  = sorted(set(tr_features[p]['taxon_id'] for p in tr_proteins))
    taxon_to_idx = {taxon: i for i, taxon in enumerate(unique_taxa)}

    # build non-PCA feature matrices
    X_tr = other_enriched_features(merged_tr_conf, tr_features, taxon_to_idx)
    X_te = other_enriched_features(merged_test_conf, te_features, taxon_to_idx)

    # Fit PCA on training proteins only, transform both splits
    PCA_DIM   = 64
    pca_model = fit_pca_on_proteins(tr_features, PCA_DIM)

    X_tr_emb = transform_pca(merged_tr_conf, tr_features, pca_model)
    X_te_emb = transform_pca(merged_test_conf, te_features, pca_model)

    # combine into complete enriched feature matrices
    X_tr_full = build_complete_feature_matrix(X_tr, X_tr_emb)
    X_te_full = build_complete_feature_matrix(X_te, X_te_emb)

    print(f'Train enriched feature matrix: {X_tr_full.shape}')
    print(f'Test enriched feature matrix : {X_te_full.shape}')