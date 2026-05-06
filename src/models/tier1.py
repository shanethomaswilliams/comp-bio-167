"""
Tier 1 ensemble methods for CAFA-5.

No-training or lightly-tuned combiners that operate on merged predictions:
    mean, max, rank_avg, weighted_mean

All methods subclass `EnsembleMethod` and expose:

    method.fit(train_data, train_labels, ia_weights) -> self
    method.predict(test_data) -> DataFrame[protein_id, GO_term, confidence]

Orchestration (CV, submission writing, logging) lives in src/main.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import fmax
from src.models.ensemble_method import EnsembleMethod


# ============================================================================
# Tier 1 methods
# ============================================================================

class Tier1Method(EnsembleMethod):
    """Tier 1 methods default to a no-op fit. Subclasses override predict()
    (and may override fit() if they have hyperparameters to tune)."""

    def fit(self, train_data, train_labels, ia_weights):
        return self


class MeanMethod(Tier1Method):
    """Arithmetic mean of confidences across models (missing = 0)."""
    name = "mean"

    def predict(self, test_data):
        out = test_data[['protein_id', 'GO_term']].copy()
        out['confidence'] = test_data[self._conf_cols(test_data)].mean(axis=1)
        return out


class MaxMethod(Tier1Method):
    """Max confidence across models — optimistic union."""
    name = "max"

    def predict(self, test_data):
        out = test_data[['protein_id', 'GO_term']].copy()
        out['confidence'] = test_data[self._conf_cols(test_data)].max(axis=1)
        return out


class RankAvgMethod(Tier1Method):
    """Per-protein rank averaging — robust to miscalibrated scores."""
    name = "rank_avg"

    @staticmethod
    def _rank_with_ties(vals):
        """Dense rank: highest value gets rank 1, ties averaged."""
        vals = np.asarray(vals, dtype=float)
        order = np.argsort(-vals)
        ranks = np.empty(len(vals), dtype=float)
        ranks[order] = np.arange(1, len(vals) + 1)
        sorted_vals = vals[order]
        i = 0
        while i < len(sorted_vals):
            j = i + 1
            while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        return ranks

    def predict(self, test_data):
        conf_cols = self._conf_cols(test_data)
        pieces = []
        for _, group in test_data.groupby('protein_id', sort=False):
            worst = len(group) + 1
            rank_cols = []
            for col in conf_cols:
                vals = group[col].values
                r = self._rank_with_ties(vals)
                # Zero-confidence predictions get the worst rank
                rank_cols.append(np.where(vals > 0, r, worst))

            mean_rank = np.mean(rank_cols, axis=0)
            out = group[['protein_id', 'GO_term']].copy()
            out['confidence'] = np.clip(
                1.0 - (mean_rank - 1.0) / worst, 0.01, 0.99
            )
            pieces.append(out)
        return pd.concat(pieces, ignore_index=True)


# ============================================================================
# Registry
# ============================================================================

TIER1_METHODS = {
    MeanMethod.name:         MeanMethod,
    MaxMethod.name:          MaxMethod,
    RankAvgMethod.name:      RankAvgMethod,
}


def build_tier1_methods(names=None):
    """Instantiate Tier 1 methods by name. None -> all of them."""
    if names is None:
        names = list(TIER1_METHODS.keys())
    return [TIER1_METHODS[n]() for n in names]