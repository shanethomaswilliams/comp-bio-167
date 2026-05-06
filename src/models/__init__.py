"""Ensemble methods organized by tier."""
from src.models.ensemble_method import EnsembleMethod

from src.models.tier1 import (
    Tier1Method,
    MeanMethod,
    MaxMethod,
    RankAvgMethod,
    TIER1_METHODS,
    build_tier1_methods,
)
from src.models.tier2 import (
    Tier2Method,
    WeightedMeanMethod,
    LogRegMethod,
    XGBoostMethod,
    MLPMethod,
    TIER2_METHODS,
    build_tier2_methods,
)

from src.models.tier3 import (
    Tier3Method,
    SoftMoEMethod,
    HardRoutingMethod,
    TIER3_METHODS,
    build_tier3_methods,
)

# Registry of all available methods across tiers. Extend as tiers 3+ are added.
ALL_METHODS = {**TIER1_METHODS, **TIER2_METHODS, **TIER3_METHODS}

# Tier -> registry map used by main.py to resolve --tier.
METHODS_BY_TIER = {
    1: TIER1_METHODS,
    2: TIER2_METHODS,
    3: TIER3_METHODS,
}

__all__ = [
    "EnsembleMethod",
    "Tier1Method",
    "MeanMethod",
    "MaxMethod",
    "RankAvgMethod",
    "Tier2Method",
    "WeightedMeanMethod",
    "LogRegMethod",
    "XGBoostMethod",
    "MLPMethod",
    "Tier3Method",
    "SoftMoEMethod",
    "HardRoutingMethod",
    "TIER1_METHODS",
    "TIER2_METHODS",
    "TIER3_METHODS"
    "ALL_METHODS",
    "METHODS_BY_TIER",
    "build_tier1_methods",
    "build_tier2_methods",
    "build_tier3_methods",
]