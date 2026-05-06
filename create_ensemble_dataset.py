"""
One-shot preprocessing: merge per-model prediction TSVs into unified ensemble
datasets AND generate cross-validation splits. Run this once when the base
predictions change; downstream training/eval reads the precomputed files.

Inputs (from config.json):
    data.<model>.train_path / test_path  — per-model prediction TSVs
    train_labels                          — ground truth GO annotations
    ia_labels                             — information accretion weights

Outputs (to --output-dir, default ./data/final_data/):
    train_merged.tsv          — protein_id, GO_term, conf_<model_*>, label
    test_merged.tsv           — protein_id, GO_term, conf_<model_*>
    train_terms.tsv           — copy of ground truth labels
    IA.txt                    — copy of IA weights
    splits/
        f{0..K-1}_split.csv      — full-pool CV folds (protein_id, split)
        f{0..K-1}_split_mf.csv   — MFO-pool CV folds
        f{0..K-1}_split_bp.csv   — BPO-pool CV folds
        f{0..K-1}_split_cc.csv   — CCO-pool CV folds
    manifest.json             — provenance: source paths, row counts, timestamps

Each split CSV is a long-format table:
    protein_id,split
    P12345,train
    Q67890,val
    ...

Usage:
    python -m src.create_ensemble_datasets --config config.json
    python -m src.create_ensemble_datasets --config config.json --output-dir data/final_data --folds 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.dataloaders import DataConfig

ASPECTS = [('MFO', 'mf'), ('BPO', 'bp'), ('CCO', 'cc')]


# ============================================================================
# Merge
# ============================================================================

def load_pred_tsv(path, conf_col):
    """Load one model's predictions. Deduplicates on (protein, term) via max."""
    df = pd.read_csv(
        path, sep='\t', header=None,
        names=['protein_id', 'GO_term', conf_col],
    )
    df = df.groupby(['protein_id', 'GO_term'], as_index=True)[conf_col].max()
    return df.to_frame()


def merge_predictions(paths, model_names):
    """Outer-join predictions from all models on (protein_id, GO_term)."""
    dfs = [load_pred_tsv(p, f'conf_{name}') for p, name in zip(paths, model_names)]
    merged = pd.concat(dfs, axis=1, join='outer').fillna(0.0).reset_index()
    return merged


def attach_labels(merged, train_terms):
    """Add binary 'label' column: 1 if (protein, term) in train_terms else 0."""
    positives = set(zip(train_terms['EntryID'], train_terms['term']))
    keys = list(zip(merged['protein_id'], merged['GO_term']))
    merged = merged.copy()
    merged['label'] = np.fromiter(
        (1.0 if k in positives else 0.0 for k in keys),
        dtype=np.float32, count=len(keys),
    )
    return merged


# ============================================================================
# Splits
# ============================================================================

def make_splits(proteins, n_folds, seed=42):
    """Run GroupKFold on an array of protein IDs.

    Returns a list of (train_proteins, val_proteins) tuples, one per fold.
    Each protein is its own group, so GroupKFold just partitions unique proteins.
    A permutation is applied first since GroupKFold does not shuffle on its own.
    """
    proteins = np.asarray(proteins)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(proteins))
    proteins_perm = proteins[perm]

    gkf = GroupKFold(n_splits=n_folds)
    folds = []
    for tr_idx, va_idx in gkf.split(proteins_perm, groups=proteins_perm):
        folds.append((proteins_perm[tr_idx], proteins_perm[va_idx]))
    return folds


def write_split_csv(train_proteins, val_proteins, path):
    """Long-format split CSV: protein_id,split."""
    rows = (
        [(p, 'train') for p in train_proteins] +
        [(p, 'val')   for p in val_proteins]
    )
    pd.DataFrame(rows, columns=['protein_id', 'split']).to_csv(path, index=False)


def _write_fold_set(proteins, n_folds, seed, splits_dir, suffix, logger):
    """Write f0..f{n-1} splits for a given protein pool."""
    if len(proteins) < n_folds:
        logger(f"  WARNING: pool has {len(proteins)} proteins < {n_folds} folds; skipping")
        return
    folds = make_splits(proteins, n_folds=n_folds, seed=seed)
    for i, (tr, va) in enumerate(folds):
        path = splits_dir / f'f{i}_split{suffix}.csv'
        write_split_csv(tr, va, path)


def generate_all_splits(train_merged, train_terms, splits_dir, n_folds, seed, logger=print):
    """Write n_folds × 4 split files: full pool + 3 per-aspect pools."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    # Full pool: every protein with merged training predictions
    full_pool = np.unique(train_merged['protein_id'].values)
    logger(f"  full pool: {len(full_pool):,} proteins")
    _write_fold_set(full_pool, n_folds, seed, splits_dir, suffix='', logger=logger)
    manifest['all'] = {'n_proteins': int(len(full_pool)), 'n_folds': n_folds}

    # Per-aspect pools: proteins with ≥1 annotation in that aspect, intersected
    # with the prediction pool. Independent GroupKFold per aspect (seed offset
    # per aspect so the three aspects' fold assignments are actually different).
    pred_pool = set(full_pool)
    for offset, (aspect_code, aspect_tag) in enumerate(ASPECTS, start=1):
        aspect_proteins = train_terms.loc[
            train_terms['aspect'] == aspect_code, 'EntryID'
        ].unique()
        aspect_pool = np.array(
            [p for p in aspect_proteins if p in pred_pool],
            dtype=object,
        )
        logger(f"  {aspect_code} pool: {len(aspect_pool):,} proteins "
               f"({len(aspect_proteins):,} in labels, intersected with predictions)")
        _write_fold_set(
            aspect_pool, n_folds, seed + offset, splits_dir,
            suffix=f'_{aspect_tag}', logger=logger,
        )
        manifest[aspect_tag] = {
            'aspect': aspect_code,
            'n_proteins': int(len(aspect_pool)),
            'n_folds': n_folds,
        }
    return manifest


# ============================================================================
# Summary
# ============================================================================

def summarize(df, name):
    out = {
        'name': name,
        'rows': len(df),
        'proteins': int(df['protein_id'].nunique()),
        'go_terms': int(df['GO_term'].nunique()),
    }
    for col in [c for c in df.columns if c.startswith('conf_')]:
        nonzero = int((df[col] > 0).sum())
        out[f'{col}_nonzero'] = nonzero
        out[f'{col}_coverage'] = round(100 * nonzero / len(df), 2)
    if 'label' in df.columns:
        pos = int(df['label'].sum())
        out['positives'] = pos
        out['positive_rate'] = round(100 * pos / len(df), 2)
    return out


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True,
                        help="JSON config with per-model paths and labels")
    parser.add_argument('--output-dir', default='data/final_data',
                        help="Where to write the merged TSVs and splits")
    parser.add_argument('--folds', type=int, default=5,
                        help="Number of CV folds to generate")
    parser.add_argument('--seed', type=int, default=42,
                        help="RNG seed for fold assignment")
    args = parser.parse_args()

    cfg = DataConfig.from_json(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models: {cfg.model_names}")
    print(f"Output dir: {out_dir}\n")

    # ---- load auxiliary ----
    print(f"Loading labels: {cfg.train_labels_path}")
    train_terms = pd.read_csv(cfg.train_labels_path, sep='\t')
    print(f"  {len(train_terms):,} annotations, "
          f"{train_terms['EntryID'].nunique():,} proteins")

    # ---- merge train ----
    print(f"\nMerging train predictions...")
    t0 = time.time()
    train_merged = merge_predictions(cfg.train_paths(), cfg.model_names)
    print(f"  merged in {time.time() - t0:.1f}s -> {len(train_merged):,} pairs")

    print(f"Attaching labels...")
    t0 = time.time()
    train_merged = attach_labels(train_merged, train_terms)
    print(f"  labeled in {time.time() - t0:.1f}s")

    train_path = out_dir / "train_merged.tsv"
    train_merged.to_csv(train_path, sep='\t', index=False)
    print(f"  wrote {train_path}")

    # ---- merge test ----
    print(f"\nMerging test predictions...")
    t0 = time.time()
    test_merged = merge_predictions(cfg.test_paths(), cfg.model_names)
    print(f"  merged in {time.time() - t0:.1f}s -> {len(test_merged):,} pairs")

    test_path = out_dir / "test_merged.tsv"
    test_merged.to_csv(test_path, sep='\t', index=False)
    print(f"  wrote {test_path}")

    # ---- copy auxiliary files ----
    for src_path, dst_name in [
        (cfg.train_labels_path, "train_terms.tsv"),
        (cfg.ia_labels_path, "IA.txt"),
    ]:
        shutil.copyfile(src_path, out_dir / dst_name)
        print(f"  copied {dst_name}")

    # ---- generate splits ----
    print(f"\nGenerating {args.folds}-fold splits...")
    splits_dir = out_dir / "splits"
    splits_manifest = generate_all_splits(
        train_merged, train_terms, splits_dir,
        n_folds=args.folds, seed=args.seed,
    )
    print(f"  wrote splits to {splits_dir}/")

    # ---- manifest for provenance ----
    manifest = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'source_config': str(Path(args.config).resolve()),
        'model_names': cfg.model_names,
        'source_paths': {
            m: {
                'train': cfg.models[m]['train_path'],
                'test':  cfg.models[m]['test_path'],
            } for m in cfg.model_names
        },
        'outputs': {
            'train':         str(train_path),
            'test':          str(test_path),
            'train_terms':   str(out_dir / "train_terms.tsv"),
            'ia_weights':    str(out_dir / "IA.txt"),
            'splits_dir':    str(splits_dir),
        },
        'splits': {
            'n_folds': args.folds,
            'seed': args.seed,
            'pools': splits_manifest,
        },
        'train_summary': summarize(train_merged, 'train'),
        'test_summary':  summarize(test_merged,  'test'),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")

    # ---- console summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, val in manifest['train_summary'].items():
        print(f"  train.{key:<28} {val}")
    for key, val in manifest['test_summary'].items():
        print(f"  test.{key:<29} {val}")
    print(f"  splits.pools                      {list(splits_manifest.keys())}")


if __name__ == "__main__":
    main()