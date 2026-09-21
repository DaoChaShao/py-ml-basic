#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/21 21:24
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   types.py
# @Desc     :

from enum import Enum, StrEnum, unique
from typing import Any

from pydantic import BaseModel, Field


@unique
class KNNMissions(StrEnum):
    CLS = "cls"
    REG = "reg"


@unique
class KNNMetrics(StrEnum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    MINKOWSKI = "minkowski"


@unique
class Languages(StrEnum):
    EN = "English"
    CN = "Chinese"


@unique
class IrisFeatures(Enum):
    SEPAL_LENGTH = ("sepal length (cm)", "花萼长度 (厘米)")
    SEPAL_WIDTH = ("sepal width (cm)", "花萼宽度 (厘米)")
    PETAL_LENGTH = ("petal length (cm)", "花瓣长度 (厘米)")
    PETAL_WIDTH = ("petal width (cm)", "花瓣宽度 (厘米)")

    def __init__(self, en_name: str, cn_name: str):
        self.EN: str = en_name
        self.CN: str = cn_name


@unique
class IrisLabels(Enum):
    SETOSA = ("setosa", "山鸢尾")
    VERSICOLOR = ("versicolor", "变色鸢尾")
    VIRGINICA = ("virginica", "维吉尼亚鸢尾")

    def __init__(self, en_name: str, cn_name: str):
        self.EN: str = en_name
        self.CN: str = cn_name


@unique
class AveStrategies(StrEnum):
    """
    Metric strategies for multi-class classification metrics.

    - WEIGHTED: Calculate metrics for each label, and find their average weighted by support (sample count).
    - MACRO: Calculate metrics for each label, and find their unweighted mean (does not take label imbalance into account).
    - MICRO: Calculate metrics globally by counting the total true positives, false negatives and false positives.
    - SAMPLES: Calculate metrics for each instance, and find their average (only meaningful for multilabel classification).
    - BINARY: Report metrics for the class specified by pos_label (used for binary classification only).
    """
    WEIGHTED = "weighted"
    MACRO = "macro"
    MICRO = "micro"
    SAMPLES = "samples"
    BINARY = "binary"


@unique
class ScoreStrategies(StrEnum):
    """
    Score strategies for classification evaluation.

    - ACCURACY: Accuracy score
    - F1_WEIGHTED: Weighted F1 score
    - F1_MACRO: Macro F1 score
    - PRECISION_WEIGHTED: Weighted precision score
    - RECALL_WEIGHTED: Weighted recall score
    - ROC_AUC_OVR: ROC AUC score for one-vs-rest
    """
    ACCURACY = "accuracy"
    F1_WEIGHTED = "f1_weighted"
    F1_MACRO = "f1_macro"
    PRECISION_WEIGHTED = "precision_weighted"
    RECALL_WEIGHTED = "recall_weighted"
    ROC_AUC_OVR = "roc_auc_ovr"


class GridSearchTunesResponse(BaseModel):
    best_params: dict[str, Any] = Field(..., description="The best parameters found during the grid search.")
    best_score: float = Field(..., description="The best score achieved during the grid search.")
