#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/25 21:34
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Machine Learning Module - Boosting Estimators
----------------------------------------------------------------
This subpackage provides ensemble learning algorithms based on
the Boosting strategy, including Gradient Boosting estimators and so on.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .adaboost import AdaBoost, HyperAdaBoost
from .gbdtree import GBDTree, HyperGBDTree
from .xgboost import HyperXGBooster

__all__ = [
    # Boosting Estimators
    "AdaBoost",
    "GBDTree",

    # Hyper-Parameters Tuning for Boosting Estimators
    "HyperAdaBoost",
    "HyperGBDTree",
    "HyperXGBooster",
]
