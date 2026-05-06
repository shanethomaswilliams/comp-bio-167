"""
Data loading for CAFA-5 ensemble meta-learning.

Two usage modes:

1. RAW MODE (for create_ensemble_datasets.py): Construct a DataConfig from the
   original per-model JSON config to drive the merge step.

2. MERGED MODE (for training/eval): Construct an EnsembleData from a directory
   containing precomputed merged TSVs and splits. The expensive outer-join and
   label attachment happen once, ahead of time.

Merged mode inputs (all under `merged_dir`):
    train_merged.tsv   — protein_id, GO_term, conf_<model_*>, label
    test_merged.tsv    — protein_id, GO_term, conf_<model_*>
    train_terms.tsv    — EntryID, term, aspect
    IA.txt             — two columns: GO_term, IA weight  (no header)
    splits/
        f{k}_split.csv      — full-pool fold k (protein_id, split)
        f{k}_split_{mf|bp|cc}.csv  — per-aspect fold k

    Optional:
        embeddings.h5  — ProtT5 embeddings, one dataset per protein_id
                         (produced by src.extract_t5_embeddings)

Two accessor families:

    Global (all training data):
        get_train_raw(), get_train_sklearn(), get_train_pytorch()
        get_test_raw(),  get_test_sklearn(),  get_test_pytorch()

    Fold-specific (filtered by a precomputed split):
        get_fold_raw(fold, split='train'|'val', aspect=None)
        get_fold_sklearn(fold, split, aspect=None)
        get_fold_pytorch(fold, split, aspect=None, batch_size=..., shuffle=...)

All sklearn/pytorch accessors accept `include_embeddings=False` (default). When
True and an embeddings file is configured, the per-row ProtT5 vector is
concatenated to the right of the confidence columns:

    X = [conf_model_0, ..., conf_model_{M-1},  emb_0, emb_1, ..., emb_{D-1}]

`aspect` is one of None (full pool), 'mf', 'bp', 'cc'. For `aspect=None`,
the split is read from `f{k}_split.csv`; otherwise from `f{k}_split_{aspect}.csv`.

Usage:
    from src.dataloaders import EnsembleData

    data = EnsembleData.from_merged_dir(
        "data/final_data",
        embeddings_path="data/final_data/embeddings.h5",  # optional
    )

    # Global — conf features only
    X, y, meta = data.get_train_sklearn()

    # Global — conf features concatenated with ProtT5 embeddings
    X, y, meta = data.get_train_sklearn(include_embeddings=True)

    # 5-fold CV loop with embeddings
    for k in range(data.n_folds):
        Xtr, ytr, _ = data.get_fold_sklearn(k, 'train', include_embeddings=True)
        Xva, yva, _ = data.get_fold_sklearn(k, 'val',   include_embeddings=True)

    # Per-aspect fold with embeddings
    Xtr, ytr, _ = data.get_fold_sklearn(0, 'train', aspect='bp', include_embeddings=True)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ASPECT_CODES = {'mf': 'MFO', 'bp': 'BPO', 'cc': 'CCO'}


# ============================================================================
# Per-model config (used only by create_ensemble_datasets.py)
# ============================================================================

@dataclass
class DataConfig:
    """Paths to raw per-model files. Built from the original JSON config."""
    models: dict
    train_labels_path: str
    ia_labels_path: str

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            cfg = json.load(f)
        return cls(
            models=cfg["data"],
            train_labels_path=cfg["train_labels"],
            ia_labels_path=cfg["ia_labels"],
        )

    @property
    def model_names(self):
        return list(self.models.keys())

    def train_paths(self):
        return [self.models[m]["train_path"] for m in self.model_names]

    def test_paths(self):
        return [self.models[m]["test_path"] for m in self.model_names]


# ============================================================================
# Helpers
# ============================================================================

def _load_ia_weights(path):
    """Load information accretion weights into {GO_term: ia} dict."""
    df = pd.read_csv(path, sep='\t', header=None, names=['term', 'ia'])
    return dict(zip(df['term'], df['ia']))


def _resolve_aspect(aspect):
    """Normalize aspect argument to a file suffix."""
    if aspect is None:
        return ''
    a = aspect.lower()
    if a in ASPECT_CODES:
        return f'_{a}'
    inv = {v.lower(): k for k, v in ASPECT_CODES.items()}
    if a in inv:
        return f'_{inv[a]}'
    raise ValueError(
        f"unknown aspect {aspect!r}; expected one of {list(ASPECT_CODES)} "
        f"or {list(ASPECT_CODES.values())}"
    )


# ============================================================================
# Top-level data object
# ============================================================================

class EnsembleData:
    """Unified interface for raw / sklearn / pytorch access to merged data.

    Reads precomputed merged TSVs and CV splits produced by
    create_ensemble_datasets.py. Optionally attaches ProtT5 embeddings from
    an HDF5 file produced by extract_t5_embeddings.py.
    """

    def __init__(
        self,
        train_path,
        test_path,
        train_terms_path,
        ia_weights_path,
        splits_dir=None,
        embeddings_path=None,
        verbose: bool = True,
    ):
        self.train_path       = Path(train_path)
        self.test_path        = Path(test_path)
        self.train_terms_path = Path(train_terms_path)
        self.ia_weights_path  = Path(ia_weights_path)
        self.splits_dir       = Path(splits_dir) if splits_dir else None
        self.embeddings_path  = Path(embeddings_path) if embeddings_path else None
        self.verbose = verbose

        if verbose:
            print(f"Loading labels from {self.train_terms_path}")
        self.train_terms = pd.read_csv(self.train_terms_path, sep='\t')
        if verbose:
            print(f"Loading IA weights from {self.ia_weights_path}")
        self.ia_weights = _load_ia_weights(self.ia_weights_path)

        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame]  = None
        self._split_cache: dict = {}  # keyed by (fold, aspect_suffix)
        self._embeddings: Optional[dict] = None  # lazy
        self._emb_dim: Optional[int] = None

    # ---- constructor helpers -------------------------------------------

    @classmethod
    def from_merged_dir(cls, merged_dir, embeddings_path=None, **kwargs):
        """Load from a directory produced by create_ensemble_datasets.py.

        If `embeddings_path` is None and <merged_dir>/embeddings.h5 exists,
        it is picked up automatically. Pass an explicit path (or False) to
        override.
        """
        d = Path(merged_dir)
        if embeddings_path is None:
            default_emb = d / "embeddings.h5"
            embeddings_path = default_emb if default_emb.exists() else None
        elif embeddings_path is False:
            embeddings_path = None
        return cls(
            train_path       = d / "train_merged.tsv",
            test_path        = d / "test_merged.tsv",
            train_terms_path = d / "train_terms.tsv",
            ia_weights_path  = d / "IA.txt",
            splits_dir       = d / "splits",
            embeddings_path  = embeddings_path,
            **kwargs,
        )

    # ---- raw pandas (global) -------------------------------------------

    def get_train_raw(self) -> pd.DataFrame:
        """Merged training predictions with binary 'label' column."""
        if self._train_df is None:
            if self.verbose:
                print(f"Reading {self.train_path}")
            self._train_df = pd.read_csv(self.train_path, sep='\t')
            if self.verbose:
                pos = int(self._train_df['label'].sum())
                print(f"  -> {len(self._train_df):,} rows, {pos:,} positives")
        return self._train_df

    def get_test_raw(self) -> pd.DataFrame:
        """Merged test predictions (no label column)."""
        if self._test_df is None:
            if self.verbose:
                print(f"Reading {self.test_path}")
            self._test_df = pd.read_csv(self.test_path, sep='\t')
            if self.verbose:
                print(f"  -> {len(self._test_df):,} rows")
        return self._test_df

    # ---- sklearn (global) ----------------------------------------------

    def get_train_sklearn(self, include_embeddings=False):
        """Return (X, y, meta) for all training data.

        X has shape (n_rows, M) with just conf features, or (n_rows, M + D)
        with embeddings concatenated on the right.
        """
        return self._df_to_sklearn(self.get_train_raw(), include_embeddings)

    def get_test_sklearn(self, include_embeddings=False):
        """Return (X, meta) for test data (no labels)."""
        df = self.get_test_raw()
        X = df[self._conf_cols(df)].to_numpy(dtype=np.float32)
        if include_embeddings:
            X_emb = self._embedding_matrix(df['protein_id'].values)
            X = np.hstack([X, X_emb])
        meta = df[['protein_id', 'GO_term']].reset_index(drop=True)
        return X, meta

    # ---- pytorch (global) ----------------------------------------------

    def get_train_pytorch(self, batch_size=4096, shuffle=True, num_workers=0,
                          include_embeddings=False):
        """DataLoader for all training data."""
        X, y, _ = self.get_train_sklearn(include_embeddings=include_embeddings)
        return self._build_loader_xy(X, y, batch_size, shuffle, num_workers)

    def get_test_pytorch(self, batch_size=4096, num_workers=0,
                         include_embeddings=False):
        """DataLoader for test predictions (no labels)."""
        X, _ = self.get_test_sklearn(include_embeddings=include_embeddings)
        return self._build_loader_x(X, batch_size, num_workers)

    # ---- fold accessors ------------------------------------------------

    def get_fold_raw(self, fold, split='train', aspect=None):
        """Slice the merged train DataFrame by a precomputed fold split.

        Args:
            fold   : int, fold index (0..n_folds-1)
            split  : 'train' or 'val'
            aspect : None (full pool) or 'mf' / 'bp' / 'cc'
        """
        proteins = self._load_split_proteins(fold, split, aspect)
        df = self.get_train_raw()
        mask = df['protein_id'].isin(proteins)
        return df.loc[mask].reset_index(drop=True)

    def get_fold_sklearn(self, fold, split='train', aspect=None,
                         include_embeddings=False):
        """Return (X, y, meta) for the requested fold slice."""
        return self._df_to_sklearn(
            self.get_fold_raw(fold, split, aspect), include_embeddings,
        )

    def get_fold_pytorch(
        self, fold, split='train', aspect=None,
        batch_size=4096, shuffle=None, num_workers=0,
        include_embeddings=False,
    ):
        """DataLoader for a fold slice. Shuffle defaults to True for train."""
        if shuffle is None:
            shuffle = (split == 'train')
        X, y, _ = self.get_fold_sklearn(
            fold, split, aspect, include_embeddings=include_embeddings,
        )
        return self._build_loader_xy(X, y, batch_size, shuffle, num_workers)

    def get_fold_terms(self, fold, split='train', aspect=None):
        """Return train_terms restricted to proteins in this fold slice.
        If aspect is specified, further restricts terms to that aspect."""
        proteins = self._load_split_proteins(fold, split, aspect)
        tt = self.train_terms
        mask = tt['EntryID'].isin(proteins)
        if aspect is not None:
            code = ASPECT_CODES[aspect.lower()]
            mask &= (tt['aspect'] == code)
        return tt.loc[mask].reset_index(drop=True)

    def _load_split_proteins(self, fold, split, aspect):
        """Read a split CSV and return the set of protein IDs for the requested side."""
        assert split in ('train', 'val'), \
            f"split must be 'train' or 'val', got {split!r}"
        if self.splits_dir is None:
            raise RuntimeError("splits_dir not configured on this EnsembleData")

        suffix = _resolve_aspect(aspect)
        cache_key = (fold, suffix)
        if cache_key not in self._split_cache:
            path = self.splits_dir / f'f{fold}_split{suffix}.csv'
            if not path.exists():
                raise FileNotFoundError(f"split file not found: {path}")
            self._split_cache[cache_key] = pd.read_csv(path)

        df = self._split_cache[cache_key]
        return set(df.loc[df['split'] == split, 'protein_id'])

    @property
    def n_folds(self):
        """Count fold files for the full-pool splits."""
        if self.splits_dir is None or not self.splits_dir.exists():
            return 0
        pattern = re.compile(r'^f(\d+)_split\.csv$')
        return sum(1 for p in self.splits_dir.iterdir() if pattern.match(p.name))

    # ---- legacy slicing helpers ----------------------------------------

    def slice_train_raw(self, proteins):
        """Return train_raw restricted to a set of protein IDs."""
        df = self.get_train_raw()
        mask = df['protein_id'].isin(proteins)
        return df.loc[mask].reset_index(drop=True)

    def slice_train_terms(self, proteins):
        """Return train_terms restricted to a set of protein IDs."""
        mask = self.train_terms['EntryID'].isin(proteins)
        return self.train_terms.loc[mask].reset_index(drop=True)

    def unique_train_proteins(self):
        """All protein IDs in the merged training predictions."""
        return self.get_train_raw()['protein_id'].unique()

    # ---- embeddings ----------------------------------------------------

    def _load_embeddings(self):
        """Read the whole HDF5 into an in-memory {protein_id: np.ndarray} map.

        For a 1024-dim model at ~150k proteins this is ~600MB; fine for any
        modern machine. If your dataset is much larger, swap this for an
        on-demand h5py reader.
        """
        if self._embeddings is not None:
            return self._embeddings
        if self.embeddings_path is None:
            raise RuntimeError(
                "include_embeddings=True but embeddings_path was not configured. "
                "Pass embeddings_path= when constructing EnsembleData, or place "
                "embeddings.h5 next to train_merged.tsv and use from_merged_dir."
            )
        import h5py
        if self.verbose:
            print(f"Loading embeddings from {self.embeddings_path}")
        with h5py.File(self.embeddings_path, 'r') as f:
            self._embeddings = {k: f[k][:].astype(np.float32) for k in f.keys()}
        any_vec = next(iter(self._embeddings.values()))
        self._emb_dim = int(any_vec.shape[0])
        if self.verbose:
            print(f"  -> {len(self._embeddings):,} proteins, dim {self._emb_dim}")
        return self._embeddings

    def embedding_matrix(self, protein_ids, return_mask=False):
        """Public: stack embeddings for a sequence of protein_ids.

        If `return_mask=True`, returns (X, mask) where mask[i] is True iff
        protein_ids[i] had a real embedding in the HDF5. Missing rows in X are
        still zero-filled.

        This is what downstream meta-learners should use when embeddings.h5
        may only contain a subset of proteins (e.g. during partial extraction).
        """
        return self._embedding_matrix(protein_ids, return_mask=return_mask)


    def _embedding_matrix(self, protein_ids, return_mask=False):
        """Stack embeddings for a sequence of protein_ids. Missing -> zeros.

        Silently tolerates missing keys; the warning is suppressed when
        return_mask=True since the caller is explicitly handling coverage.
        """
        emb = self._load_embeddings()
        dim = self._emb_dim
        out  = np.zeros((len(protein_ids), dim), dtype=np.float32)
        mask = np.zeros(len(protein_ids), dtype=bool)
        for i, pid in enumerate(protein_ids):
            vec = emb.get(pid)
            if vec is not None:
                out[i] = vec
                mask[i] = True
        n_missing = int((~mask).sum())
        if n_missing and self.verbose and not return_mask:
            # Only warn on the legacy call path. Hybrid Tier 3 callers pass
            # return_mask=True and are expected to handle coverage themselves.
            unique_missing = sorted(set(
                pid for pid, present in zip(protein_ids, mask) if not present
            ))
            print(f"  WARNING: {n_missing:,} rows ({len(unique_missing):,} "
                f"unique proteins) missing from embeddings; filled with zeros. "
                f"First few: {unique_missing[:5]}")
        if return_mask:
            return out, mask
        return out

    # ---- internals ------------------------------------------------------

    def _df_to_sklearn(self, df, include_embeddings=False):
        """Convert the merged train DataFrame into (X, y, meta)."""
        X = df[self._conf_cols(df)].to_numpy(dtype=np.float32)
        if include_embeddings:
            X_emb = self._embedding_matrix(df['protein_id'].values)
            X = np.hstack([X, X_emb])
        y = df['label'].to_numpy(dtype=np.float32)
        meta = df[['protein_id', 'GO_term']].reset_index(drop=True)
        return X, y, meta

    @staticmethod
    def _build_loader_xy(X, y, batch_size, shuffle, num_workers):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        idx = np.arange(len(X), dtype=np.int64)
        ds = TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(idx),
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=False,
        )

    @staticmethod
    def _build_loader_x(X, batch_size, num_workers):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        idx = np.arange(len(X), dtype=np.int64)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(idx))
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        )

    @staticmethod
    def _conf_cols(df):
        return [c for c in df.columns if c.startswith('conf_')]

    @property
    def model_names(self):
        """Infer from column names in train file (conf_<n> -> <n>)."""
        df = self.get_train_raw()
        return [c[len('conf_'):] for c in self._conf_cols(df)]

    @property
    def emb_dim(self):
        """Embedding dimension. Triggers an HDF5 load if not cached."""
        if self._emb_dim is None:
            self._load_embeddings()
        return self._emb_dim