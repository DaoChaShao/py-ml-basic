#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/25 21:33
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Machine Learning Module - Bagging Estimators
----------------------------------------------------------------
This subpackage provides ensemble learning algorithms based on
the Bagging strategy, including Random Forest estimators and so on.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .forest import RandomForest

__all__ = [
    "RandomForest",
]
