#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/19 21:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Machine Learning Module - Comprehensive Toolkit
----------------------------------------------------------------
This module provides a comprehensive suite of machine learning
estimators, data preprocessing utilities, model evaluation,
and pipeline diagnostics.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .postprocessor import (
    diagnose_cls_fit,
    diagnose_reg_fit,
)
from .preprocessor import (
    FeaturesNormaliser,
    FeaturesRobustScaler,
    FeaturesStandardiser,
    FileCategories,
    FileLoader,
    NumpySeed,
    calc_labels_weight,
    chebyshev_distance,
    encode_labels,
    euclidean_distance,
    expand_polynomial_features,
    get_cls_labels_distribution,
    get_reg_labels_distribution,
    grid_search_tunes,
    manhattan_distance,
    minkowski_distance,
    split_data,
    summary_dataframe,
    tune_optimal_cls_degree,
    tune_optimal_reg_degree,
)
from .types import (
    AlphaCategories,
    AveStrategies,
    CaliforniaFeatures,
    ClsScoreStrategies,
    GridSearchTunesResponse,
    IrisFeatures,
    IrisLabels,
    KNNMetrics,
    Languages,
    LogisticSolvers,
    Missions,
    RegLosses,
    RegPenalties,
    RegScoreStrategies,
    TitanicFeatures,
    TitanicLabels,
    TreeClsCriteria,
    TreeRegCriteria,
    TreeSplitters,
)

__all__ = [
    # Postprocessing & Diagnostics
    "diagnose_cls_fit",
    "diagnose_reg_fit",

    # Preprocessing & Utilities
    "FeaturesNormaliser",
    "FeaturesRobustScaler",
    "FeaturesStandardiser",
    "FileCategories",
    "FileLoader",
    "NumpySeed",
    "calc_labels_weight",
    "chebyshev_distance",
    "encode_labels",
    "euclidean_distance",
    "expand_polynomial_features",
    "get_cls_labels_distribution",
    "get_reg_labels_distribution",
    "grid_search_tunes",
    "manhattan_distance",
    "minkowski_distance",
    "split_data",
    "summary_dataframe",
    "tune_optimal_cls_degree",
    "tune_optimal_reg_degree",

    # Preprocessing & Utilities
    "AlphaCategories",
    "AveStrategies",
    "CaliforniaFeatures",
    "ClsScoreStrategies",
    "GridSearchTunesResponse",
    "IrisFeatures",
    "IrisLabels",
    "KNNMetrics",
    "Languages",
    "LogisticSolvers",
    "Missions",
    "RegLosses",
    "RegPenalties",
    "RegScoreStrategies",
    "TitanicFeatures",
    "TitanicLabels",
    "TreeClsCriteria",
    "TreeRegCriteria",
    "TreeSplitters",
]
