#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/26 22:38
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Machine Learning Module - Linear Estimators
----------------------------------------------------------------
This subpackage provides linear machine learning estimators,
including OLS, Ridge, Lasso, Elastic Net, and SGD estimators.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .enet import ElasticNetRegressor, HyperElasticNetRegressor
from .lasso import HyperLassoRegressor, LassoRegressor
from .ols import HyperOLSRegressor, OLSRegressor
from .ridge import HyperRidgeRegressor, RidgeRegressor
from .sgd import HyperSGDRegressor, SGDReg

__all__ = [
    # Linear Estimators
    "ElasticNetRegressor",
    "LassoRegressor",
    "OLSRegressor",
    "RidgeRegressor",
    "SGDReg",

    # Hyper-Parameters Tuning for Linear Estimators
    "HyperElasticNetRegressor",
    "HyperLassoRegressor",
    "HyperOLSRegressor",
    "HyperRidgeRegressor",
    "HyperSGDRegressor",
]
