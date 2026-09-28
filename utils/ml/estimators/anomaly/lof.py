#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/28 19:26
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   lof.py
# @Desc     :

from typing import Annotated, Any, Literal, Self, override

from access_modifiers import protectedmethod
from numpy import asarray
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor

from ...types import DistanceMetrics, LOFAlgorithms
from ..base import Base


class HyperLOF(Base, BaseEstimator):
    """ Local Outlier Factor for anomaly detection. """

    @validate_call
    def __init__(
            self,
            n_neighbors: int = Field(20, gt=0),
            *,
            algorithm: str | LOFAlgorithms | Literal["auto", "ball_tree", "kd_tree", "brute"] = LOFAlgorithms.AUTO,
            leaf_size: int = Field(30, gt=0),
            p: float = Field(2.0, gt=0),
            metric: str | DistanceMetrics | Literal[
                "chebyshev", "euclidean", "manhattan", "minkowski",
            ] = DistanceMetrics.MINKOWSKI,
            contamination: str | Annotated[float, Field(gt=0, le=0.5)] | Literal["auto"] = "auto",
            novelty: bool = True,
    ) -> None:
        """
        Initialise the Local Outlier Factor estimator.

        :param n_neighbors: Number of neighboring samples.
        :param algorithm: Algorithm used to compute nearest neighbors.
        :param leaf_size: Leaf size passed to the tree algorithms.
        :param p: Power parameter for the Minkowski metric.
        :param metric: Distance metric.
        :param contamination: Expected proportion of anomalies.
        :param novelty: If True, the estimator can be used for new novelty detection.
        :return: None
        """
        super().__init__()
        self.n_neighbors: int = n_neighbors
        self.algorithm: LOFAlgorithms = LOFAlgorithms(algorithm)
        self.leaf_size: int = leaf_size
        self.p: float = p
        self.metric: DistanceMetrics = DistanceMetrics(metric)
        self.contamination: str | float | Literal["auto"] = contamination
        self.novelty: bool = novelty

    @protectedmethod
    def _init_model(self) -> None:
        """ Initialise the Local Outlier Factor model. """
        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm.value,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric.value,
            contamination=self.contamination,
            novelty=self.novelty,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """ Return estimator parameters. """
        return {
            "n_neighbors": self.n_neighbors,
            "algorithm": self.algorithm.value,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric.value,
            "contamination": self.contamination,
        }

    @override
    def set_params(self, **params: Any) -> Self:
        """ Set estimator parameters. """
        if not params:
            return self

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}.")

            if key == "algorithm":
                value = LOFAlgorithms(value)
            elif key == "metric":
                value = DistanceMetrics(value)

            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series | None = None) -> Self:
        """ Fit the Local Outlier Factor model. """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")

        self._model.fit(asarray(features))
        self._fitted = True
        return self

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict whether samples are normal or anomalous.

        Returns:
            1  -> normal
            -1 -> anomaly
        :param features: Test features.
        :return: Predictions.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.predict(asarray(features))

    def decision_function(self, features: DataFrame) -> Any:
        """
        Return shifted opposite LOF scores.

        Positive values generally indicate normal samples,
        while negative values indicate anomalous samples.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.decision_function(asarray(features))

    def score_samples(self, features: DataFrame) -> Any:
        """ Return opposite LOF scores. Lower scores indicate more abnormal samples. """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.score_samples(asarray(features))

    def score(self, features: DataFrame) -> float:
        """
        Return the mean anomaly score of the input samples.

        :param features: Test features.
        :return: Mean anomaly score.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return float(self._model.score_samples(asarray(features)).mean())
