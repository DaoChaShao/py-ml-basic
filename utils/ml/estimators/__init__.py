#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 12:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Machine Learning Module - Estimators Suite
----------------------------------------------------------------
This subpackage provides the core Machine Learning algorithm
implementations, built upon a unified base protocol (`Base`).
Includes KNN, Ordinary Least Squares (OLS), Stochastic Gradient
Descent (SGD), Ridge (L2), and Lasso (L1) estimators.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .bagging import HyperRandomForest, RandomForest
from .base import Base
from .bayes import NaiveBayes
from .boosting import (
    AdaBoost,
    GBDTree,
    HyperAdaBoost,
    HyperGBDTree,
    HyperXGBooster,
)
from .cart import DecisionTree, HyperDecisionTree
from .kmeans import HyperKMeans
from .knn import KNN, HyperKNN
from .linear import ElasticNetRegressor, HyperElasticNetRegressor
from .linear.lasso import HyperLassoRegressor, LassoRegressor
from .linear.ols import HyperOLSRegressor, OLSRegressor
from .linear.ridge import HyperRidgeRegressor, RidgeRegressor
from .linear.sgd import HyperSGDRegressor, SGDRegressor
from .logistic import HyperLogisticRegClassifier, LogisticRegClassifier

__all__ = [
    # Protocols
    "Base",

    # Estimators
    "DecisionTree",
    "ElasticNetRegressor",
    "KNN",
    "LassoRegressor",
    "LogisticRegClassifier",
    "NaiveBayes",
    "OLSRegressor",
    "RidgeRegressor",
    "SGDRegressor",
    # Hyper-Parameters Estimators
    "HyperDecisionTree",
    "HyperElasticNetRegressor",
    "HyperKMeans",
    "HyperKNN",
    "HyperLassoRegressor",
    "HyperLogisticRegClassifier",
    "HyperOLSRegressor",
    "HyperRidgeRegressor",
    "HyperSGDRegressor",

    # Boosting Estimators
    "AdaBoost",
    "GBDTree",
    # Hyper-Parameters Tuning for Boosting Estimators
    "HyperAdaBoost",
    "HyperGBDTree",
    "HyperXGBooster",

    # Bagging Estimators
    "RandomForest",
    # Hyper-Parameters Tuning for Bagging Estimators
    "HyperRandomForest",
]
