#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/27 17:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Machine Learning Module - Clustering Estimators
----------------------------------------------------------------
This subpackage provides cluster algorithm implementations,
including K-Means and Gaussian Mixture Models (GMM), together
with their hyper-parameter tuning estimators.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .gmm import HyperGaussianMixture
from .kmeans import HyperKMeans

__all__ = [
    # Clustering Estimators

    # Hyper-Parameters Clustering Estimators
    "HyperGaussianMixture",
    "HyperKMeans",
]
