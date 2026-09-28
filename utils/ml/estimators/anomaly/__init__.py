#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/28 00:43
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Machine Learning Module - Hyper-Parameters Estimators
----------------------------------------------------------------
This subpackage provides hyper-parameterized machine learning
estimators, including Isolation Forest estimators and so on.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .forest import HyperIsolationForest

__all__ = [
    # Hyper-Parameters Estimators
    "HyperIsolationForest"
]
