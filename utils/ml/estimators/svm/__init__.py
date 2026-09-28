#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/28 00:45
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Machine Learning Module - Support Vector Machine Estimators
----------------------------------------------------------------
This subpackage provides support vector machine estimators,
including SVC and SVR estimators.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .svc import HyperSVClassifier
from .svr import HyperSVRegressor

__all__ = [
    # Support Vector Machine Estimators
    "HyperSVClassifier",
    "HyperSVRegressor",
]
