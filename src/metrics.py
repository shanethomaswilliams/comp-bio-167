"""
CAFA-5 evaluation metric: weighted F-max.

The official metric is the arithmetic mean of F-max across the three ontology
aspects (MFO, BPO, CCO), using information-accretion-weighted precision/recall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ASPECTS = ('MFO', 'BPO', 'CCO')


def fmax(preds_df, ground_truth_df, ia_weights, thresholds=None):
    """Weighted F-max averaged across MFO, BPO, CCO.

    Args:
        preds_df        : DataFrame[protein_id, GO_term, confidence]
        ground_truth_df : DataFrame[EntryID, term, aspect]
        ia_weights      : dict[GO_term -> ia]
        thresholds      : array of cutoffs to sweep; defaults to 0.01..0.99 step 0.02

    Returns:
        dict with per-aspect F-max and 'mean'
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.02)

    # {protein: {aspect: set(terms)}}
    true_map = {}
    for _, row in ground_truth_df.iterrows():
        true_map.setdefault(row['EntryID'], {}) \
                .setdefault(row['aspect'], set()).add(row['term'])

    # {protein: {term: confidence}}
    pred_map = (
        preds_df.groupby('protein_id')
        .apply(lambda g: dict(zip(g['GO_term'], g['confidence'])))
        .to_dict()
    )

    scores = {}
    for aspect in ASPECTS:
        gt_proteins = [p for p in true_map if aspect in true_map[p]]
        if not gt_proteins:
            scores[aspect] = 0.0
            continue
        best_f = 0.0
        for t in thresholds:
            sp_num = sp_den = sr_num = sr_den = 0.0
            for p in gt_proteins:
                true_terms = true_map[p].get(aspect, set())
                pred_terms = {g for g, c in pred_map.get(p, {}).items() if c >= t}
                tp_ia   = sum(ia_weights.get(g, 0.0) for g in true_terms & pred_terms)
                pre_den = sum(ia_weights.get(g, 0.0) for g in pred_terms)
                rec_den = sum(ia_weights.get(g, 0.0) for g in true_terms)
                sp_num += tp_ia; sp_den += pre_den if pre_den > 0 else 1e-9
                sr_num += tp_ia; sr_den += rec_den if rec_den > 0 else 1e-9
            prec = sp_num / sp_den
            rec  = sr_num / sr_den
            f = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            if f > best_f:
                best_f = f
        scores[aspect] = best_f

    scores['mean'] = float(np.mean([scores[a] for a in ASPECTS]))
    return scores