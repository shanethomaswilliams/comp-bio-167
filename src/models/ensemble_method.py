
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import fmax


# ============================================================================
# Base class
# ============================================================================

class EnsembleMethod:
    """Abstract base for ensemble methods.

    Subclasses must set `name` and implement `fit` and `predict`.
    Subclasses may override `__init__` for hyperparameters.
    """
    name: str = "base"

    def __init__(self):
        pass

    def fit(self, train_data, train_labels, ia_weights):
        """Fit the method on merged training predictions.

        Args:
            train_data   : DataFrame[protein_id, GO_term, conf_*, label]
            train_labels : DataFrame[EntryID, term, aspect] for this data
            ia_weights   : dict[GO_term -> information accretion weight]

        Returns:
            self
        """
        raise NotImplementedError

    def predict(self, test_data) -> pd.DataFrame:
        """Produce per-(protein, term) confidences from merged predictions.

        Args:
            test_data : DataFrame[protein_id, GO_term, conf_*]

        Returns:
            DataFrame[protein_id, GO_term, confidence]
        """
        raise NotImplementedError

    @staticmethod
    def _conf_cols(df):
        return [c for c in df.columns if c.startswith('conf_')]