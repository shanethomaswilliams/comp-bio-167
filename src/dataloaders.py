# Shared data loaders for CAFA-5 integration (Tier 1, 2, 3, 4)

from pathlib import Path
from typing import Union

# numpy not used yet
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# constants
SEED = 42
ASPECTS = ("MFO", "BPO", "CCO")
_TERMS_COLS  = {"EntryID", "term", "aspect"}
_IA_KEYS_MIN = 50  # floor for IA.txt entry count
# double check floor


def load_prediction_file(filepath: Union[str, Path], model_index: int) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Prediction file not found: {filepath}  (cwd: {Path.cwd()})")

    conf_col = f"conf_{model_index}"
    # header = none because tsv has no column names
    df = pd.read_csv(filepath, sep="\t", header=None,
                     names=["protein_id", "GO_term", conf_col])

    if df.empty:
        raise ValueError(f"Prediction file is empty: {filepath}")

    if df[conf_col].isnull().any():
        n = df[conf_col].isnull().sum()
        raise ValueError(f"{n} missing confidence values in {filepath.name} — check column count")

    bad = df[(df[conf_col] < 0.0) | (df[conf_col] > 1.0)]
    if not bad.empty:
        raise ValueError(f"{len(bad)} confidence values outside [0,1] in {filepath.name}")

    print(f"  Loaded {len(df):,} rows from '{filepath.name}' as '{conf_col}'")
    return df


def load_merged_predictions(filepaths: list[Union[str, Path]]) -> pd.DataFrame:
    if not filepaths:
        raise ValueError("filepaths list is empty — need at least one file")

    dfs = []
    for i, fp in enumerate(filepaths):
        df = load_prediction_file(fp, model_index=i)
        dfs.append(df.set_index(["protein_id", "GO_term"]))  # index for aligned join

    # outer join so we don't drop pairs missing from one model, fill with 0
    merged = pd.concat(dfs, axis=1, join="outer").fillna(0.0).reset_index()
    print(f"  Merged {len(filepaths)} model(s) -> {len(merged):,} (protein, GO_term) pairs")
    return merged


def load_train_terms(filepath: Union[str, Path]) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"train_terms not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t")

    missing = _TERMS_COLS - set(df.columns)
    if missing:
        raise ValueError(f"train_terms.tsv missing columns: {missing} (found: {list(df.columns)})")

    unknown = set(df["aspect"].unique()) - set(ASPECTS)
    if unknown:
        print(f"  WARNING: unexpected aspect values in train_terms: {unknown}")

    print(f"  Loaded {len(df):,} annotations for {df['EntryID'].nunique():,} proteins")
    return df


def load_ia_weights(filepath: Union[str, Path]) -> dict[str, float]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"IA weights file not found: {filepath}")

    ia_df = pd.read_csv(filepath, sep="\t", header=None, names=["term", "ia"])

    if len(ia_df) < _IA_KEYS_MIN:
        raise ValueError(f"IA file has only {len(ia_df)} entries — may be truncated")

    ia_weights = dict(zip(ia_df["term"], ia_df["ia"]))
    print(f"  Loaded IA weights for {len(ia_weights):,} GO terms")
    return ia_weights


def split_fit_eval(merged_conf, train_terms, labels=None, test_size=0.15, seed=SEED):
    # group by protein so the same protein doesn't end up in both splits
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    fit_idx, eval_idx = next(splitter.split(merged_conf, groups=merged_conf['protein_id']))

    merged_fit  = merged_conf.iloc[fit_idx].reset_index(drop=True)
    merged_eval = merged_conf.iloc[eval_idx].reset_index(drop=True)

    fit_proteins  = set(merged_fit['protein_id'])
    eval_proteins = set(merged_eval['protein_id'])

    fit_terms  = train_terms[train_terms['EntryID'].isin(fit_proteins)].reset_index(drop=True)
    eval_terms = train_terms[train_terms['EntryID'].isin(eval_proteins)].reset_index(drop=True)

    # slice labels provided (For Tier 2, not Tier 1)
    labels_fit  = labels.iloc[fit_idx].reset_index(drop=True) if labels is not None else None
    labels_eval = labels.iloc[eval_idx].reset_index(drop=True) if labels is not None else None

    return merged_fit, merged_eval, fit_terms, eval_terms, labels_fit, labels_eval


def load_tier1_data(
    train_files: list[Union[str, Path]],
    test_files:  list[Union[str, Path]],
    data_dir:    Union[str, Path] = "cafa-5-protein-function-prediction/",
) -> dict:
    data_dir = Path(data_dir)

    def resolve(f):
        p = Path(f)
        return p if p.is_absolute() else data_dir / p  # support absolute or relative paths

    print("\n=== Loading Tier 1 data ===")
    merged_train = load_merged_predictions([resolve(f) for f in train_files])
    merged_test  = load_merged_predictions([resolve(f) for f in test_files])
    train_terms  = load_train_terms(data_dir / "train_terms.tsv")
    ia_weights   = load_ia_weights(data_dir / "IA.txt")
    merged_fit, merged_eval, fit_terms, eval_terms, _, _ = split_fit_eval(merged_train, train_terms)
    print("=== Data loading complete ===\n")

    return {
        "merged_train": merged_train,
        "merged_test":  merged_test,
        "train_terms":  train_terms,
        "ia_weights":   ia_weights,
        "merged_fit":   merged_fit,
        "merged_eval":  merged_eval,
        "fit_terms":    fit_terms,
        "eval_terms":   eval_terms,
    }


def build_labels(merged_conf, train_terms):
    # 1 if protein has GO term, 0 if not
    true_pairs = set(zip(train_terms['EntryID'], train_terms['term']))
    labels = merged_conf[['protein_id', 'GO_term']].copy()
    labels['label'] = [
        int((row.protein_id, row.GO_term) in true_pairs)
        for row in labels.itertuples(index=False)
    ]
    return labels


def load_tier2_data(train_files, test_files,
                    data_dir="cafa-5-protein-function-prediction/"):
    data_dir = Path(data_dir)
    def resolve(f):
        p = Path(f)
        return p if p.is_absolute() else data_dir / p

    merged_train = load_merged_predictions([resolve(f) for f in train_files])
    merged_test  = load_merged_predictions([resolve(f) for f in test_files])
    train_terms  = load_train_terms(data_dir / "train_terms.tsv")
    ia_weights   = load_ia_weights(data_dir / "IA.txt")

    # build binary labels before split
    labels = build_labels(merged_train, train_terms)

    merged_fit, merged_eval, fit_terms, eval_terms, labels_fit, labels_eval = split_fit_eval(
        merged_train, train_terms, labels=labels
    )

    return {
        "merged_train": merged_train,
        "merged_test":  merged_test,
        "train_terms":  train_terms,
        "ia_weights":   ia_weights,
        "labels":       labels,
        "merged_fit":   merged_fit,
        "merged_eval":  merged_eval,
        "fit_terms":    fit_terms,
        "eval_terms":   eval_terms,
        "labels_fit":   labels_fit,
        "labels_eval":  labels_eval,
    }
