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
algorithms and data preprocessing utilities, including
classifiers, distance metrics, file loading, label encoding,
feature scaling, data splitting, and class weight computation.
****************************************************************
"""

from .base import MetricStrategies, Base

from .knn import (
    KNNMissions, KNNMetrics,
    KNN,
    euclidean_distance, manhattan_distance, chebyshev_distance, minkowski_distance,
)
from .preprocessor import (
    FileCat,
    NumpySeed,
    FileLoader,
    get_labels_distribution, encode_labels,
    split_data,
    FeaturesNormaliser, FeaturesStandardiser,
    calc_labels_weight,
)

from importlib.metadata import metadata, PackageNotFoundError

try:
    _meta = metadata("py-ml-basic")
    __version__ = _meta.get("Version", "0.0.0")
    __author__ = _meta.get("Author", "Shawn Yu")
except PackageNotFoundError:
    __author__ = "Shawn Yu"
    __version__ = "0.1.0"

__all__ = [
    "MetricStrategies", "Base",

    "KNNMissions", "KNNMetrics",
    "KNN",
    "euclidean_distance", "manhattan_distance", "chebyshev_distance", "minkowski_distance",

    "FileCat",
    "NumpySeed",
    "FileLoader",
    "get_labels_distribution", "encode_labels",
    "split_data",
    "FeaturesNormaliser", "FeaturesStandardiser",
    "calc_labels_weight",
]
