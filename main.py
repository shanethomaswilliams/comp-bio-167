"""
CAFA-5 ensemble runner.

Usage (from project root):
    # Tier 1 — no-training combiners
    python -m src.main --tier 1 --method mean

    # Tier 2 — learned flat combiners
    python -m src.main --tier 2 --method logreg
    python -m src.main --tier 2 --method xgb --tune --run-name tuned_xgb

    # Tier 2 + ProtT5 embeddings ("Enriched Stacking")
    python -m src.main --tier 2 --method logreg --use-embeddings --emb-pca 128

    # Tier 3 — routing-based combiners (Soft MoE / Hard Routing)
    python -m src.main --tier 3 --method soft_moe --fallback max
    python -m src.main --tier 3 --method soft_moe --tune --run-name tuned_soft_moe
    python -m src.main --tier 3 --method hard_routing

The --tune flag grid-searches hyperparameters defined in src/finetune.py,
picks the config with highest CV F-max, and uses it for the final test
submission.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.dataloaders import EnsembleData
from src.metrics import fmax
from src.models.tier1 import MeanMethod, MaxMethod, RankAvgMethod
from src.models.tier2 import (
    WeightedMeanMethod, LogRegMethod, XGBoostMethod, MLPMethod,
)

N_FOLDS = 5
MAX_TERMS_PER_PROTEIN = 1500


# ============================================================================
# Logging
# ============================================================================

def setup_logger(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "logs.txt"

    logger = logging.getLogger("cafa5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')

    fh = logging.FileHandler(log_path, mode='a'); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

    logger.info(f"Logging to {log_path}")
    return logger


# ============================================================================
# Submission writer
# ============================================================================

def write_submission(preds_df, out_path, logger):
    sub = preds_df[preds_df['confidence'] > 0.0].copy()
    sub['confidence'] = sub['confidence'].round(3)
    sub = sub[sub['confidence'] > 0.0]
    sub = (
        sub.sort_values(['protein_id', 'confidence'], ascending=[True, False])
           .groupby('protein_id').head(MAX_TERMS_PER_PROTEIN)
    )
    sub.to_csv(out_path, sep='\t', header=False, index=False)
    logger.info(f"  wrote {len(sub):,} predictions -> {out_path}")


# ============================================================================
# fit/predict dispatch
# ============================================================================

def call_fit(model, train_df, train_tms, ia_weights, embeddings):
    if embeddings is not None:
        model.fit(train_df, train_tms, ia_weights, embeddings=embeddings)
    else:
        model.fit(train_df, train_tms, ia_weights)
    return model


def call_predict(model, df, embeddings):
    if embeddings is not None:
        return model.predict(df, embeddings=embeddings)
    return model.predict(df)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CAFA-5 ensemble runner.")
    parser.add_argument('--data-dir', default='data/final_data')
    parser.add_argument('--tier', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--method', required=True,
                        help="Tier 1: mean|max|rank_avg.  "
                             "Tier 2: weighted_mean|logreg|xgb|mlp.  "
                             "Tier 3: soft_moe|hard_routing.")
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--tune', action='store_true',
                        help="Grid-search hyperparameters using grids in "
                             "src/finetune.py. Supported for Tier 2 & Tier 3.")
    parser.add_argument('--device', default='auto',
                        help="Device for torch-based methods: auto/cpu/cuda/mps.")

    # Embedding-related flags
    parser.add_argument('--embeddings-path', default=None,
                        help="Path to embeddings.h5. If omitted, auto-detected "
                             "at <data-dir>/embeddings.h5. Required for Tier 3 "
                             "or Tier 2 --use-embeddings.")
    parser.add_argument('--use-embeddings', action='store_true',
                        help="Tier 2 only: concatenate ProtT5 embeddings with "
                             "conf features (Enriched Stacking).")
    parser.add_argument('--emb-pca', type=int, default=None,
                        help="Tier 2 --use-embeddings only: reduce embedding "
                             "dim via PCA to this many components.")
    parser.add_argument('--fallback', choices=['max', 'mean'], default='max',
                        help="Tier 3 only: fill-in rule for rows whose protein "
                             "has no embedding (partial extraction).")

    args = parser.parse_args()

    # ---- setup ------------------------------------------------------------
    results_dir = Path(args.results_dir)
    if args.run_name:
        results_dir = results_dir / args.run_name
    logger = setup_logger(results_dir)

    needs_embeddings = (args.tier == 3) or (args.tier == 2 and args.use_embeddings)

    load_kwargs = dict(verbose=False)
    if needs_embeddings:
        load_kwargs['embeddings_path'] = args.embeddings_path
    data = EnsembleData.from_merged_dir(args.data_dir, **load_kwargs)

    embeddings_arg = data if needs_embeddings else None

    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Base models: {data.model_names}")
    logger.info(f"Tier {args.tier}, method: {args.method}")
    if needs_embeddings:
        logger.info(f"Embeddings: {data.embeddings_path}")
        if args.tier == 2:
            logger.info(f"  use_embeddings=True, emb_pca={args.emb_pca}")
        if args.tier == 3:
            logger.info(f"  fallback='{args.fallback}'")
    logger.info(f"Device: {args.device}")

    device_kw = {} if args.device == 'auto' else {'device': args.device}

    # ========================================================================
    # TIER 1
    # ========================================================================
    if args.tier == 1:
        if   args.method == 'mean':     model = MeanMethod()
        elif args.method == 'max':      model = MaxMethod()
        elif args.method == 'rank_avg': model = RankAvgMethod()
        else: raise ValueError(f"unknown tier 1 method: {args.method}")

        logger.info(f"\n=== {args.method} (Tier 1) ===")

        fold_scores = []
        for i in range(N_FOLDS):
            val_df  = data.get_fold_raw(fold=i, split='val')
            val_tms = data.get_fold_terms(fold=i, split='val')

            t0 = time.time()
            preds = model.predict(val_df)
            score = fmax(preds, val_tms, data.ia_weights)
            elapsed = time.time() - t0

            fold_scores.append(score)
            logger.info(
                f"  fold {i+1}/{N_FOLDS}: "
                f"MFO={score['MFO']:.4f}  BPO={score['BPO']:.4f}  "
                f"CCO={score['CCO']:.4f}  mean={score['mean']:.4f}  ({elapsed:.1f}s)"
            )

        for key in ('MFO', 'BPO', 'CCO', 'mean'):
            vals = np.array([s[key] for s in fold_scores])
            logger.info(f"  CV {key:<4}: {vals.mean():.4f} ± {vals.std():.4f}")

        test_preds = model.predict(data.get_test_raw())
        write_submission(
            test_preds, results_dir / f"{args.method}_test_submission.tsv", logger,
        )
        return

    # ========================================================================
    # TIER 2 / TIER 3 — learned combiners (with optional tuning)
    # ========================================================================

    def build_model(**kwargs):
        if args.tier == 2:
            emb_kw = {}
            if args.use_embeddings:
                emb_kw['use_embeddings'] = True
                if args.emb_pca is not None:
                    emb_kw['emb_pca'] = args.emb_pca
            all_kw = {**emb_kw, **kwargs}
            if args.method == 'mlp':
                all_kw = {**device_kw, **all_kw}
            if   args.method == 'weighted_mean': return WeightedMeanMethod(**all_kw)
            elif args.method == 'logreg':        return LogRegMethod(**all_kw)
            elif args.method == 'xgb':           return XGBoostMethod(**all_kw)
            elif args.method == 'mlp':           return MLPMethod(**all_kw)
            else: raise ValueError(f"unknown tier 2 method: {args.method}")
        else:  # tier == 3
            from src.models.tier3 import SoftMoEMethod, HardRoutingMethod
            all_kw = {'fallback': args.fallback, **device_kw, **kwargs}
            if   args.method == 'soft_moe':     return SoftMoEMethod(**all_kw)
            elif args.method == 'hard_routing': return HardRoutingMethod(**all_kw)
            else: raise ValueError(f"unknown tier 3 method: {args.method}")

    logger.info(f"\n=== {args.method} (Tier {args.tier}) ===")

    # ---- Tuning path ------------------------------------------------------
    if args.tune:
        from src.finetune import TUNE_FUNCTIONS
        if args.method not in TUNE_FUNCTIONS:
            raise ValueError(
                f"--tune is not implemented for method '{args.method}'. "
                f"Available: {sorted(TUNE_FUNCTIONS.keys())}"
            )

        # Non-tuned constructor args to merge into every grid config.
        extra = {}
        if args.tier == 3:
            extra['fallback'] = args.fallback
        if device_kw and args.method in ('mlp', 'soft_moe', 'hard_routing'):
            extra.update(device_kw)
        if args.tier == 2 and args.use_embeddings:
            extra['use_embeddings'] = True
            if args.emb_pca is not None:
                extra['emb_pca'] = args.emb_pca

        tune_fn = TUNE_FUNCTIONS[args.method]
        tune_result = tune_fn(
            data, N_FOLDS, logger,
            extra_kwargs=extra or None,
            embeddings=embeddings_arg,
        )
        tune_result.log_leaderboard(logger)

        best_config = tune_result.best_config
        fold_scores = tune_result.best_fold_scores
        logger.info(f"  using best config for refit: {best_config}")

    # ---- Default path (no tune) -----------------------------------------
    else:
        best_config = {}
        fold_scores = []
        for i in range(N_FOLDS):
            train_df  = data.get_fold_raw(fold=i, split='train')
            val_df    = data.get_fold_raw(fold=i, split='val')
            train_tms = data.get_fold_terms(fold=i, split='train')
            val_tms   = data.get_fold_terms(fold=i, split='val')

            t0 = time.time()
            model = build_model()
            call_fit(model, train_df, train_tms, data.ia_weights, embeddings_arg)
            preds = call_predict(model, val_df, embeddings_arg)
            score = fmax(preds, val_tms, data.ia_weights)
            elapsed = time.time() - t0

            fold_scores.append(score)
            logger.info(
                f"  fold {i+1}/{N_FOLDS}: "
                f"MFO={score['MFO']:.4f}  BPO={score['BPO']:.4f}  "
                f"CCO={score['CCO']:.4f}  mean={score['mean']:.4f}  ({elapsed:.1f}s)"
            )

    # ---- CV summary (shared) ---------------------------------------------
    logger.info(f"\n  --- CV summary ({args.method}) ---")
    for key in ('MFO', 'BPO', 'CCO', 'mean'):
        vals = np.array([s[key] for s in fold_scores])
        logger.info(f"  CV {key:<4}: {vals.mean():.4f} ± {vals.std():.4f}")

    # ---- Final refit + test submission -----------------------------------
    logger.info(f"\n  refitting on full training set...")
    t0 = time.time()
    final_model = build_model(**best_config)
    call_fit(
        final_model, data.get_train_raw(), data.train_terms,
        data.ia_weights, embeddings_arg,
    )
    logger.info(f"  refit complete in {time.time() - t0:.1f}s")

    test_preds = call_predict(final_model, data.get_test_raw(), embeddings_arg)
    write_submission(
        test_preds, results_dir / f"{args.method}_test_submission.tsv", logger,
    )


if __name__ == "__main__":
    main()