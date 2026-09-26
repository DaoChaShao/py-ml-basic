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

from .bagging import RandomForest, HyperRandomForest
from .base import Base
from .boosting import (
    AdaBoost,
    GBDTree,
    HyperAdaBoost,
    HyperGBDTree,
    HyperXGBooster,
)
from .cart import DecisionTree, HyperDecisionTree
from .enet import ElasticNetRegressor, HyperElasticNetRegressor
from .knn import KNN, HyperKNN
from .lasso import LassoRegressor, HyperLassoRegressor
from .logistic import LogisticRegClassifier, HyperLogisticRegClassifier
from .ols import OLSRegressor, HyperOLSRegressor
from .ridge import RidgeRegressor, HyperRidgeRegressor
from .sgd import SGDRegressor, HyperSGDRegressor

__all__ = [
    # Protocols
    "Base",

    # Estimators
    "DecisionTree",
    "ElasticNetRegressor",
    "KNN",
    "LassoRegressor",
    "LogisticRegClassifier",
    "OLSRegressor",
    "RidgeRegressor",
    "SGDRegressor",
    # Hyper-Parameters Estimators
    "HyperDecisionTree",
    "HyperElasticNetRegressor",
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
