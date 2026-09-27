#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/27 17:15
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   gmm.py
# @Desc     :

from typing import Any, Literal, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture

from utils.ml import GMMCovarianceCategories, GMMInitParamsCategories
from utils.ml.estimators import Base


class HyperGaussianMixture(Base, BaseEstimator):
    """ HyperGaussianMixture is a class for hyperparameter tuning of the Gaussian Mixture Model algorithm. """

    @validate_call
    def __init__(
            self,
            n_components: int,
            *,
            covariance: str | GMMCovarianceCategories | Literal[
                "full", "tied", "diag", "spherical"
            ] = GMMCovarianceCategories.FULL,
            epochs: int = Field(100, gt=1),
            init_params: str | GMMInitParamsCategories | Literal[
                "kmeans", "random", "random_from_data", "k-means++"
            ] = GMMInitParamsCategories.KMEANS,
            randomness: int | None = 27,
    ) -> None:
        """
        Initialise the HyperGaussianMixture estimator.

        :param n_components: Number of mixture components.
        :param covariance: Type of covariance parameters.
        :param epochs: Maximum number of EM iterations.
        :param init_params: Method used to initialise the weights, means and covariances.
        :param randomness: Random seed.
        :return: None
        """
        super().__init__()
        self.n_components: int = n_components
        self.covariance: str = covariance
        self.epochs: int = epochs
        self.init_params: str = init_params
        self.randomness: int = randomness

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying GaussianMixture model.

        :return: None
        """
        self._model = SklearnGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance,
            max_iter=self.epochs,
            init_params=self.init_params,
            random_state=self.randomness,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        This method allows sklearn utilities such as GridSearchCV to inspect
        and clone the estimator.

        :param deep: Whether to return parameters of nested estimators.
        :return: Estimator parameters.
        """
        return {
            "n_components": self.n_components,
            "covariance": self.covariance,
            "epochs": self.epochs,
            "init_params": self.init_params,
            "randomness": self.randomness,
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
                    f"Invalid parameter {key!r} for HyperGaussianMixture. "
                    f"Valid parameters are: {list(valid_params)}."
                )
            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series | None = None) -> Self:
        """
        Fit the Gaussian Mixture Model.

        :param features: The features of the training data.
        :param labels: The labels of the training data.
        :return: The estimator.
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
        Predict the component labels for the input data.

        :param features: The features of the input data.
        :return: Predicted component labels.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")

        return self._model.predict(features)

    def predict_proba(self, features: DataFrame) -> Any:
        """
        Predict posterior probabilities of each component.

        :param features: The features of the input data.
        :return: Posterior probabilities.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")

        return self._model.predict_proba(features)

    def score(self, features: DataFrame) -> float:
        """
        Return the average log-likelihood of the input data.

        :param features: The features of the input data.
        :return: Average log-likelihood.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")

        return float(self._model.score(features))

    def aic(self, features: DataFrame) -> float:
        """ Return the Akaike information criterion for the input data. """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return float(self._model.aic(features))

    def bic(self, features: DataFrame) -> float:
        """ Return the Bayesian information criterion for the input data. """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return float(self._model.bic(features))
