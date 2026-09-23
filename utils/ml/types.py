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
class FileCategories(StrEnum):
    CSV = "csv"
    EXCEL = "excel"


@unique
class Missions(StrEnum):
    """ Machine learning missions. """
    CLS = "cls"
    REG = "reg"


@unique
class KNNMetrics(StrEnum):
    """ K-Nearest Neighbors metrics. """
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    MINKOWSKI = "minkowski"


@unique
class Languages(StrEnum):
    """ Programming languages. """
    EN = "English"
    CN = "Chinese"


@unique
class IrisFeatures(Enum):
    """ Iris dataset features. """
    SEPAL_LENGTH = ("sepal length (cm)", "花萼长度 (厘米)")
    SEPAL_WIDTH = ("sepal width (cm)", "花萼宽度 (厘米)")
    PETAL_LENGTH = ("petal length (cm)", "花瓣长度 (厘米)")
    PETAL_WIDTH = ("petal width (cm)", "花瓣宽度 (厘米)")

    def __init__(self, en_name: str, cn_name: str):
        self.EN: str = en_name
        self.CN: str = cn_name


@unique
class IrisLabels(Enum):
    """ Iris dataset labels. """
    SETOSA = ("setosa", "山鸢尾")
    VERSICOLOR = ("versicolor", "变色鸢尾")
    VIRGINICA = ("virginica", "维吉尼亚鸢尾")

    def __init__(self, en_name: str, cn_name: str) -> None:
        """
        Initialize the enum member with English and Chinese names.

        :param en_name: English name
        :param cn_name: Chinese name
        :return: None
        """
        super().__init__()
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
class ClsScoreStrategies(StrEnum):
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


@unique
class RegScoreStrategies(StrEnum):
    """
    Metric strategies for regression evaluation.

    - RMSE: Root Mean Squared Error. Standard regression metric, penalises larger errors more heavily.
    - MSE: Mean Squared Error. Average squared difference between predicted and actual values.
    - MAE: Mean Absolute Error. Average absolute difference, more robust to outliers than MSE.
    - R2: Coefficient of determination ($R^2$). Represents the proportion of variance explained by the model (best = 1.0).
    - MAPE: Mean Absolute Percentage Error. Relative error percentage, useful for scale-independent comparisons.
    """
    RMSE = "neg_root_mean_squared_error"
    MSE = "neg_mean_squared_error"
    MAE = "neg_mean_absolute_error"
    R2 = "r2"
    MAPE = "neg_mean_absolute_percentage_error"


class GridSearchTunesResponse(BaseModel):
    """ Response model for grid search tunes. """
    best_params: dict[str, Any] = Field(..., description="The best parameters found during the grid search.")
    best_score: float = Field(..., description="The best score achieved during the grid search.")


@unique
class CaliforniaFeatures(Enum):
    """ California housing dataset features. """
    MED_INC = ("MedInc", "该区域居民收入中位数", "万美金")
    HOUSE_AGE = ("HouseAge", "房屋工龄中位数", "年")
    AVE_ROOMS = ("AveRooms", "平均房间数", "间")
    AVE_BEDRMS = ("AveBedrms", "平均卧室数", "间")
    POPULATION = ("Population", "该区域人口数", "人")
    AVE_OCCUP = ("AveOccup", "平均每户入住人数", "人")
    LATITUDE = ("Latitude", "纬度", "度")
    LONGITUDE = ("Longitude", "经度", "度")

    def __init__(self, en_name: str, cn_name: str, unit: str) -> None:
        """
        Initialize the enum member with English name, Chinese name, and unit.

        :param en_name: English name
        :param cn_name: Chinese name
        :param unit: Unit of measurement
        """
        super().__init__()
        self.EN: str = en_name
        self.CN: str = cn_name
        self.UNIT: str = unit


@unique
class RegLosses(StrEnum):
    """ Supported loss functions for SGDRegressor in Scikit-Learn. """
    SQUARED_ERROR = "squared_error"
    ABSOLUTE_ERROR = "absolute_error"  # Not for SGD Regression
    HUBER = "huber"
    EPSILON_INSENSITIVE = "epsilon_insensitive"
    SQUARED_EPSILON_INSENSITIVE = "squared_epsilon_insensitive"


@unique
class AlphaCategories(StrEnum):
    """ Supported learning rate schedules for Regressor. """
    INVSCALING = "invscaling"
    OPTIMAL = "optimal"
    CONSTANT = "constant"
    ADAPTIVE = "adaptive"


@unique
class RegPenalties(StrEnum):
    """ Supported regularisation penalties for SGDRegressor and linear models. """
    L2 = "l2"
    L1 = "l1"
    ELASTICNET = "elasticnet"


@unique
class LogisticSolvers(StrEnum):
    """ Supported solvers for LogisticRegression in Scikit-Learn. """
    LBFGS = "lbfgs"
    SAGA = "saga"
    SAG = "sag"
    LIBLINEAR = "liblinear"
    NEWTON_CG = "newton-cg"
    NEWTON_CHOLESKY = "newton-cholesky"


@unique
class TitanicFeatures(Enum):
    """ Titanic raw dataset features (13 columns). """
    PCLASS = ("pclass", "客舱等级", "Social-economic class (1 = 1st, 2 = 2nd, 3 = 3rd)")
    NAME = ("name", "乘客姓名", "Passenger name")
    SEX = ("sex", "性别", "Biological sex (male / female)")
    AGE = ("age", "年龄", "Age in years")
    SIBSP = ("sibsp", "同辈亲属数", "Number of siblings / spouses aboard")
    PARCH = ("parch", "直系亲属数", "Number of parents / children aboard")
    TICKET = ("ticket", "船票编号", "Ticket number")
    FARE = ("fare", "乘客票价", "Passenger fare")
    CABIN = ("cabin", "客舱号码", "Cabin number")
    EMBARKED = ("embarked", "登船港口", "Port of embarkation (C, Q, S)")
    BOAT = ("boat", "救生艇编号", "Lifeboat number (if rescued)")
    BODY = ("body", "遗体编号", "Body identification number (if deceased)")
    HOME_DEST = ("home.dest", "家乡/目的地", "Home / Destination")

    def __init__(self, en_name: str, cn_name: str, desc: str = "") -> None:
        """
        Initialise the enum member with English name, Chinese name, and description.

        :param en_name: English name
        :param cn_name: Chinese name
        :param desc: Description
        """
        super().__init__()
        self.EN: str = en_name
        self.CN: str = cn_name
        self.DESC: str = desc

    @classmethod
    def get_en_names(cls) -> list[str]:
        """ Get all original English feature names. """
        return [feature.EN for feature in cls]

    @classmethod
    def get_cn_names(cls) -> list[str]:
        """ Get all Chinese feature names. """
        return [feature.CN for feature in cls]


@unique
class TitanicLabels(Enum):
    """ Titanic dataset target label (survived). """
    DECEASED = ("0", "Perished", "遇难", 0)
    SURVIVED = ("1", "Survived", "幸存", 1)

    def __init__(self, str_label: str, en_name: str, cn_name: str, int_label: int) -> None:
        """
        Initialise the enum member with string, Chinese, and integer values.

        :param str_label: String value
        :param en_name: English name
        :param cn_name: Chinese name
        :param int_label: Integer value
        :return: None
        """
        super().__init__()
        self.STR_LABEL: str = str_label
        self.EN: str = en_name
        self.CN: str = cn_name
        self.INT_LABEL: int = int_label

    @classmethod
    def get_str_labels(cls) -> list[str]:
        """ Get target string labels ['0', '1']. """
        return [target.STR_VAL for target in cls]

    @classmethod
    def get_int_labels(cls) -> list[int]:
        """ Get target integer labels [0, 1]. """
        return [target.INT_VAL for target in cls]


@unique
class TreeClsCriteria(StrEnum):
    GINI = "gini"
    ENTROPY = "entropy"
    LOG_LOSS = "log_loss"


@unique
class TreeRegCriteria(StrEnum):
    SQUARED_ERROR = "squared_error"
    FRIEDMAN_MSE = "friedman_mse"
    ABSOLUTE_ERROR = "absolute_error"
    POISSON = "poisson"


@unique
class TreeSplitters(StrEnum):
    BEST = "best"
    RANDOM = "random"
