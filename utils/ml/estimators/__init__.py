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

from .bagging import RandomForest
from .base import Base
from .boosting import GBDTree
from .cart import DecisionTree
from .enet import ElasticNetReg
from .knn import KNN
from .lasso import LassoReg
from .logistic import LogisticRegClassifier
from .ols import OLSReg
from .ridge import RidgeReg
from .sgd import SGDReg

__all__ = [
    # Protocols
    "Base",

    # Estimators
    "DecisionTree",
    "ElasticNetReg",
    "GBDTree",
    "KNN",
    "LassoReg",
    "LogisticRegClassifier",
    "OLSReg",
    "RandomForest",
    "RidgeReg",
    "SGDReg",

]
