#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/19 21:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   knn.py
# @Desc     :

from typing import Any, Literal, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .base import Base
from .types import KNNMetrics, Missions


class KNN(Base):
    """ KNN class for classification and regression. """

    def __init__(
            self,
            mission: str | Missions | Literal["cls", "reg"],
            *,
            n_neighbours: int = 5,
            metric: str | KNNMetrics | Literal[
                "euclidean", "manhattan", "chebyshev", "minkowski"
            ] = KNNMetrics.MINKOWSKI,
            p: float = 2.0,
    ):
        """
        Initialise the KNN class.

        :param mission: The mode of the KNN algorithm ("cls" or "reg").
        :param n_neighbours: The number of neighbours to consider.
        :param metric: Distance metric to use.
        :param p: Power parameter for the Minkowski metric. (Only effective when metric='minkowski')
        """
        super().__init__()
        self._mission: Missions = Missions(mission)
        self._neighbours: int = n_neighbours
        self._metric: KNNMetrics = KNNMetrics(metric)
        self._p: float = p
        self._fitted: bool = False

        self._init_model()

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the KNN estimator based on the mode.

        :return: None
        """
        _estimator = KNeighborsClassifier if self._mission is Missions.CLS else KNeighborsRegressor
        self._model = _estimator(n_neighbors=self._neighbours, metric=self._metric.value, p=self._p)

    @override
    def train(self, features: DataFrame, labels: Series) -> None:
        """
        Train the KNN estimator with the given features and labels.

        :param features: The features to train the KNN estimator.
        :param labels: The labels to train the KNN estimator.
        :return: None
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")
        self._model.fit(features, labels)
        self._fitted = True

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict the labels for the given features using the KNN estimator.

        :param features: The features to predict the labels for.
        :return: The predicted labels.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    def confidence(self, features: DataFrame) -> Any:
        """
        Predict the probabilities for the given features using the KNN estimator.

        :param features: The features to predict the probabilities for.
        :return: The predicted probabilities.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        if self._mission is not Missions.CLS:
            raise RuntimeError("Confidence probabilities are only available for classification.")
        return self._model.predict_proba(features)

    @property
    def mission(self) -> Missions:
        """
        Get the mode of the KNN algorithm.

        :return: The mode of the KNN algorithm.
        """
        return self._mission

    @property
    def neighbours(self) -> int:
        """
        Get the number of neighbors considered for classification or regression.

        :return: The number of neighbors considered for classification or regression.
        """
        return self._neighbours

    @property
    def metric(self) -> KNNMetrics:
        """
        Get the metric used for distance calculation.

        :return: The metric used for distance calculation.
        """
        return self._metric

    @property
    def p(self) -> float:
        """
        Get the value of p used for the Minkowski distance metric.

        :return: The value of p used for the Minkowski distance metric.
        """
        return self._p

    def __repr__(self) -> str:
        """
        Return a string representation of the KNN object.

        :return: A string representation of the KNN object.
        """
        return (
            f"KNN("
            f"mission={self._mission.value!r}, "
            f"neighbours={self._neighbours!r}, "
            f"metric={self._metric.value!r}, "
            f"p={self._p!r}"
            f")"
        )


def euclidean_distance(x1: Any, x2: Any) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param x1: The first point.
    :param x2: The second point.
    :return: The Euclidean distance between the two points.
    """
    return sum((a - b) ** 2 for a, b in zip(x1, x2, strict=True)) ** 0.5


def manhattan_distance(x1: Any, x2: Any) -> float:
    """
    Calculate the Manhattan distance between two points.

    :param x1: The first point.
    :param x2: The second point.
    :return: The Manhattan distance between the two points.
    """
    return sum(abs(a - b) for a, b in zip(x1, x2, strict=True))


def chebyshev_distance(x1: Any, x2: Any) -> float:
    """
    Calculate the Chebyshev distance between two points.

    :param x1: The first point.
    :param x2: The second point.
    :return: The Chebyshev distance between the two points.
    """
    return max(abs(a - b) for a, b in zip(x1, x2, strict=True))


def minkowski_distance(x1: Any, x2: Any, p: float) -> float:
    """
    Calculate the Minkowski distance between two points.
    - If p = 1, it becomes the Manhattan distance.
    - If p = 2, it becomes the Euclidean distance.
    - If p = infinity, it becomes the Chebyshev distance.

    :param x1: The first point.
    :param x2: The second point.
    :param p: The order of the Minkowski distance.
    :return: The Minkowski distance between the two points.
    """
    if p < 1:
        raise ValueError("p must be greater than or equal to 1.")
    return sum(abs(a - b) ** p for a, b in zip(x1, x2, strict=True)) ** (1 / p)


if __name__ == "__main__":
    pass
