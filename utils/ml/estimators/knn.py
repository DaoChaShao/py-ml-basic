#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/19 21:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   knn.py
# @Desc     :

from typing import Any, Literal, override, Self

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from utils.ml.types import KNNMetrics, Missions

from .base import Base


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

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the KNN estimator based on the mode.

        :return: None
        """
        _estimator = KNeighborsClassifier if self._mission is Missions.CLS else KNeighborsRegressor
        self._model = _estimator(n_neighbors=self._neighbours, metric=self._metric.value, p=self._p)

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
        Train the KNN estimator with the given features and labels.

        :param features: The features to train the KNN estimator.
        :param labels: The labels to train the KNN estimator.
        :return: The estimator
        """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")

        self._model.fit(features, labels)
        self._fitted = True
        return self

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
            f"p={self._p!r})"
        )


class HyperKNN(Base, BaseEstimator):
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
        self.mission: Missions = Missions(mission)
        self.n_neighbours: int = n_neighbours
        self.metric: KNNMetrics = KNNMetrics(metric)
        self.p: float = p

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the KNN estimator based on the mode.

        :return: None
        """
        _estimator = KNeighborsClassifier if self.mission is Missions.CLS else KNeighborsRegressor
        self._model = _estimator(n_neighbors=self.n_neighbours, metric=self.metric.value, p=self.p)

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        This method allows sklearn utilities such as GridSearchCV to inspect and clone the estimator.

        :param deep: Whether to return parameters of nested estimators.
        :return: Estimator parameters.
        """
        return {
            "mission": self.mission,
            "n_neighbours": self.n_neighbours,
            "metric": self.metric,
            "p": self.p,
        }

    @override
    def set_params(self, **params: Any) -> Self:
        """
        This method is required by sklearn's hyperparameter search utilities.

        :param params: Parameters to set.
        :return: The estimator with parameters set.
        """
        if not params:
            return self

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for HyperKNN. "
                    f"Valid parameters are: {list(valid_params)}."
                )
            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
        Train the KNN estimator with the given features and labels.

        :param features: The features to train the KNN estimator.
        :param labels: The labels to train the KNN estimator.
        :return: The estimator
        """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")

        self._model.fit(features, labels)
        self._fitted = True
        return self

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
        if self.mission is not Missions.CLS:
            raise RuntimeError("Confidence probabilities are only available for classification.")
        return self._model.predict_proba(features)
