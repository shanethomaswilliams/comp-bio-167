"""
Hyperparameter tuning for Tier 2 and Tier 3 meta-learners.

How it works:
    For each hyperparameter config in a method's grid, run the full K-fold CV
    on precomputed splits and compute mean F-max across folds. Pick the config
    with the highest mean F-max.

Grids are defined at the top of this file — edit them to tune a wider or
narrower range. Each method has its own `tune_*` function with a fixed grid;
all delegate to `tune_generic` which does the actual search.

Usage (from main.py):
    from src.finetune import tune_logreg
    result = tune_logreg(data, n_folds=5, logger=logger)
    best_model = LogRegMethod(**result.best_config)

    # Tier 3 — pass the EnsembleData as embeddings to route through to fit/predict
    from src.finetune import tune_soft_moe
    result = tune_soft_moe(data, n_folds=5, logger=logger, embeddings=data)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.metrics import fmax
from src.models.tier2 import (
    WeightedMeanMethod, LogRegMethod, XGBoostMethod, MLPMethod,
)
from src.models.tier3 import SoftMoEMethod, HardRoutingMethod


# ============================================================================
# HYPERPARAMETER GRIDS — TIER 2
# ============================================================================

# weighted_mean has no hyperparameters (NNLS is fully determined).
WEIGHTED_MEAN_GRID = {}

LOGREG_GRID = {
    'C':            [0.01, 0.1, 1.0, 10.0],
    'class_weight': [None, 'balanced'],
    'max_iter':     [1000],
}

XGB_GRID = {
    'n_estimators':  [100, 200, 400],
    'max_depth':     [3, 4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample':     [0.9],
    'random_state':  [42],
}

MLP_GRID = {
    'hidden':       [(64, 32), (128, 64)],
    'dropout':      [0.2, 0.3],
    'lr':           [1e-3],
    'weight_decay': [1e-5],
    'batch_size':   [4096],
    'epochs':       [50],
    'patience':     [5],
    'random_state': [42],
}


# ============================================================================
# HYPERPARAMETER GRIDS — TIER 3
# ============================================================================
# Tier 3 models are PyTorch-based and each config trains K times across folds.
# Keep grids small by default. The knob that matters most for Soft MoE is
# entropy_reg (prevents gate collapse); for Hard Routing it's min_annotations
# (filters noisy "best team" labels from proteins with few annotations).

SOFT_MOE_GRID = {
    'hidden':        [(128, 64), (256, 128)],
    'dropout':       [0.3],
    'entropy_reg':   [0.0, 0.01, 0.1],
    'lr':            [1e-3],
    'weight_decay':  [1e-5],
    'batch_size':    [4096],
    'epochs':        [30],
    'patience':      [5],
    'random_state':  [42],
}
# Product: 2 × 1 × 3 × 1 × 1 × 1 × 1 × 1 × 1 = 6 configs × K folds

HARD_ROUTING_GRID = {
    'hidden':           [(64, 32), (128, 64), (256, 128)],
    'dropout':          [0.2, 0.3],
    'min_annotations':  [3, 5, 10],
    'lr':               [1e-3],
    'weight_decay':     [1e-5],
    'batch_size':       [512],
    'epochs':           [50],
    'patience':         [5],
    'random_state':     [42],
}
# Product: 3 × 2 × 3 × 1 × 1 × 1 × 1 × 1 × 1 = 18 configs × K folds
# (hard routing is fast — trains on proteins, not rows)


# ============================================================================
# Results container
# ============================================================================

@dataclass
class TuneResult:
    method_name:       str
    best_config:       dict
    best_cv_score:     dict
    best_fold_scores:  list
    all_configs:       list = field(default_factory=list)

    def log_leaderboard(self, logger, top_n=None):
        rows = sorted(self.all_configs, key=lambda r: -r['cv_score']['mean'])
        if top_n is not None:
            rows = rows[:top_n]
        logger.info(f"  {self.method_name} tuning leaderboard ({len(rows)} configs):")
        for r in rows:
            logger.info(
                f"    mean={r['cv_score']['mean']:.4f}  "
                f"MFO={r['cv_score']['MFO']:.4f} BPO={r['cv_score']['BPO']:.4f} "
                f"CCO={r['cv_score']['CCO']:.4f}   {r['config']}"
            )


# ============================================================================
# Core search
# ============================================================================

def _expand_grid(grid):
    """Dict of lists -> list of dicts. Empty grid returns [{}]."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def _cv_score(model_cls, config, data, n_folds, logger=None, log_prefix="",
              embeddings=None):
    """Run K-fold CV for a single hyperparameter config.

    If `embeddings` is not None, it is passed as a keyword argument to
    fit() and predict() (required for Tier 3 and for Tier 2 use_embeddings).
    """
    fold_scores = []
    for i in range(n_folds):
        train_df  = data.get_fold_raw(fold=i, split='train')
        val_df    = data.get_fold_raw(fold=i, split='val')
        train_tms = data.get_fold_terms(fold=i, split='train')
        val_tms   = data.get_fold_terms(fold=i, split='val')

        t0 = time.time()
        model = model_cls(**config)
        if embeddings is not None:
            model.fit(train_df, train_tms, data.ia_weights, embeddings=embeddings)
            preds = model.predict(val_df, embeddings=embeddings)
        else:
            model.fit(train_df, train_tms, data.ia_weights)
            preds = model.predict(val_df)
        score = fmax(preds, val_tms, data.ia_weights)
        elapsed = time.time() - t0

        fold_scores.append(score)
        if logger is not None:
            logger.info(
                f"    {log_prefix}fold {i+1}/{n_folds}: "
                f"MFO={score['MFO']:.4f}  BPO={score['BPO']:.4f}  "
                f"CCO={score['CCO']:.4f}  mean={score['mean']:.4f}  ({elapsed:.1f}s)"
            )

    agg = {
        k: float(np.mean([s[k] for s in fold_scores]))
        for k in ('MFO', 'BPO', 'CCO', 'mean')
    }
    return agg, fold_scores


def tune_generic(method_name, model_cls, grid, data, n_folds, logger,
                 extra_kwargs=None, embeddings=None):
    """Run the full hyperparameter search for any Tier 2 or Tier 3 method.

    Args:
        method_name  : string label for logging
        model_cls    : the method subclass
        grid         : dict[str -> list] (can be empty)
        data         : EnsembleData instance
        n_folds      : number of CV folds
        logger       : logging.Logger
        extra_kwargs : dict of kwargs merged into every config. Used for
                       constructor-time args that aren't being tuned —
                       e.g. {'fallback': 'max', 'device': 'cuda'} for Tier 3.
                       Grid values override these if the same key appears in both.
        embeddings   : if not None (e.g. an EnsembleData instance), passed to
                       fit/predict as a keyword argument. Required for Tier 3.

    Returns:
        TuneResult with best config + leaderboard.
    """
    extra_kwargs = extra_kwargs or {}
    configs = _expand_grid(grid)
    logger.info(f"\n--- Tuning {method_name} ({len(configs)} configs × {n_folds} folds) ---")
    if extra_kwargs:
        logger.info(f"  extra kwargs applied to every config: {extra_kwargs}")
    if embeddings is not None:
        logger.info(f"  passing embeddings to fit/predict")

    all_configs = []
    for idx, cfg in enumerate(configs, start=1):
        full_cfg = {**extra_kwargs, **cfg}
        logger.info(f"  [{idx}/{len(configs)}] config = {cfg}")
        agg, fold_scores = _cv_score(
            model_cls, full_cfg, data, n_folds,
            logger=logger, log_prefix="  ", embeddings=embeddings,
        )
        logger.info(
            f"    -> CV mean F-max = {agg['mean']:.4f}  "
            f"(MFO={agg['MFO']:.4f} BPO={agg['BPO']:.4f} CCO={agg['CCO']:.4f})"
        )
        all_configs.append({
            'config':       cfg,
            'full_config':  full_cfg,
            'cv_score':     agg,
            'fold_scores':  fold_scores,
        })

    best = max(all_configs, key=lambda r: r['cv_score']['mean'])
    result = TuneResult(
        method_name       = method_name,
        best_config       = best['config'],
        best_cv_score     = best['cv_score'],
        best_fold_scores  = best['fold_scores'],
        all_configs       = all_configs,
    )
    logger.info(
        f"  best {method_name}: mean F-max = {best['cv_score']['mean']:.4f}  "
        f"config = {best['config']}"
    )
    return result


# ============================================================================
# Per-method wrappers — TIER 2
# ============================================================================

def tune_weighted_mean(data, n_folds, logger, extra_kwargs=None, embeddings=None):
    return tune_generic(
        'weighted_mean', WeightedMeanMethod, WEIGHTED_MEAN_GRID,
        data, n_folds, logger, extra_kwargs, embeddings,
    )


def tune_logreg(data, n_folds, logger, extra_kwargs=None, embeddings=None):
    return tune_generic(
        'logreg', LogRegMethod, LOGREG_GRID,
        data, n_folds, logger, extra_kwargs, embeddings,
    )


def tune_xgb(data, n_folds, logger, extra_kwargs=None, embeddings=None):
    return tune_generic(
        'xgb', XGBoostMethod, XGB_GRID,
        data, n_folds, logger, extra_kwargs, embeddings,
    )


def tune_mlp(data, n_folds, logger, extra_kwargs=None, embeddings=None):
    return tune_generic(
        'mlp', MLPMethod, MLP_GRID,
        data, n_folds, logger, extra_kwargs, embeddings,
    )


# ============================================================================
# Per-method wrappers — TIER 3
# ============================================================================

def tune_soft_moe(data, n_folds, logger, extra_kwargs=None, embeddings=None):
    """Tune SoftMoEMethod over hidden sizes and entropy_reg.

    entropy_reg is the key knob — too low lets the gate collapse to one team,
    too high forces uniform routing (equivalent to mean).
    """
    if embeddings is None:
        raise ValueError(
            "tune_soft_moe requires embeddings; pass embeddings=<EnsembleData>"
        )
    return tune_generic(
        'soft_moe', SoftMoEMethod, SOFT_MOE_GRID,
        data, n_folds, logger, extra_kwargs, embeddings,
    )


def tune_hard_routing(data, n_folds, logger, extra_kwargs=None, embeddings=None):
    """Tune HardRoutingMethod over hidden sizes and min_annotations.

    min_annotations controls noise on the "best team" labels: proteins with
    few annotations produce argmax labels that are essentially random.
    """
    if embeddings is None:
        raise ValueError(
            "tune_hard_routing requires embeddings; pass embeddings=<EnsembleData>"
        )
    return tune_generic(
        'hard_routing', HardRoutingMethod, HARD_ROUTING_GRID,
        data, n_folds, logger, extra_kwargs, embeddings,
    )


# Dispatch table for main.py
TUNE_FUNCTIONS = {
    'weighted_mean': tune_weighted_mean,
    'logreg':        tune_logreg,
    'xgb':           tune_xgb,
    'mlp':           tune_mlp,
    'soft_moe':      tune_soft_moe,
    'hard_routing':  tune_hard_routing,
}