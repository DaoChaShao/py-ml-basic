#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/26 22:46
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   kmeans.py
# @Desc     :

from typing import Any, Literal, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from ..types import KMeansAlgorithms, KMeansInitCategories
from .base import Base


class HyperKMeans(Base, BaseEstimator):
    """ HyperKMeans is a class for hyperparameter tuning of the KMeans algorithm. """

    @validate_call
    def __init__(
            self,
            n_clusters: int,
            *,
            init_cat: str | KMeansInitCategories | Literal[
                "k-means++", "random"
            ] = KMeansInitCategories.K_MEANS_PLUS_PLUS,
            epochs: int = Field(300, gt=1),
            randomness: int = 27,
            algorithm: str | KMeansAlgorithms | Literal["lloyd", "elkan"] = KMeansAlgorithms.LLOYD
    ) -> None:
        """
        Initialise the HyperKMeans estimator.

        :param n_clusters: Number of clusters.
        :param init_cat: Initialization method.
        :param epochs: Number of epochs.
        :param randomness: Random seed.
        :param algorithm: Algorithm to use.
        :return: None
        """
        super().__init__()
        self.n_clusters: int = n_clusters
        self.init_cat: KMeansInitCategories = KMeansInitCategories(init_cat)
        self.epochs: int = epochs
        self.randomness: int = randomness
        self.algorithm: KMeansAlgorithms = KMeansAlgorithms(algorithm)

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying KMeans model.

        :return: None
        """
        self._model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init_cat.value,
            max_iter=self.epochs,
            random_state=self.randomness,
            algorithm=self.algorithm.value,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        This method allows sklearn utilities such as GridSearchCV to inspect and clone the estimator.

        :param deep: Whether to return parameters of nested estimators.
        :return: Estimator parameters.
        """
        return {
            "n_clusters": self.n_clusters,
            "init_cat": self.init_cat,
            "epochs": self.epochs,
            "randomness": self.randomness,
            "algorithm": self.algorithm,
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
                    f"Invalid parameter {key!r} for HyperKMeans. "
                    f"Valid parameters are: {list(valid_params)}."
                )
            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series | None = None) -> Self:
        """
       Fit the KMeans model. sklearn estimators conventionally return self from fit().

       :param features: The features of the training data.
       :param labels: The labels of the training data.
       :return: The estimator
       """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")

        self._model.fit(features)
        self._fitted = True
        return self

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict labels or regression values.

        :param features: The features of the input data.
        :return: Predicted labels or regression values.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.predict(features)

    def score(self, features: DataFrame, labels: Series, ) -> float:
        """
        Return the default sklearn score.

        :param features: The features of the input data.
        :param labels: The labels of the input data.
        :return: The score.
        """

        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return float(self._model.score(features, labels))
