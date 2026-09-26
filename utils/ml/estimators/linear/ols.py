#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 19:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   ols.py
# @Desc     :   Ordinary Least Squares

from typing import Any, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from utils.ml.estimators.base import Base


class OLSRegressor(Base):

    def __init__(self, is_intercept: bool = True) -> None:
        """
        Initialise the Ordinary Least Squares estimator.

        :param is_intercept: Whether to include an intercept term in the model. Default is True.
        :return: None
        """
        super().__init__()
        self._is_intercept: bool = is_intercept

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the estimator based on the mode.

        :return: None
        """
        self._model = LinearRegression(fit_intercept=self._is_intercept)

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
       Train the estimator with the given features and labels.

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
        Predict the labels for the given features using the estimator.

        :param features: The features to predict the labels for.
        :return: The predicted labels.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    @property
    def coefficient(self) -> Any:
        """
        Get the regression coefficients (weights).

        :return: The regression coefficients (weights).
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.coef_

    @property
    def intercept(self) -> Any:
        """
        Get the regression intercept (bias).

        :return: The regression intercept (bias).
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        if not self._is_intercept:
            return None
        return self._model.intercept_

    def __repr__(self) -> str:
        """
        Get the string representation of the estimator.

        :return: The string representation of the estimator.
        """
        return f"Linear(is_intercept={self._is_intercept})"


class HyperOLSRegressor(Base, BaseEstimator):

    def __init__(self, is_intercept: bool = True) -> None:
        """
        Initialise the Ordinary Least Squares estimator.

        :param is_intercept: Whether to include an intercept term in the model. Default is True.
        :return: None
        """
        super().__init__()
        self.is_intercept: bool = is_intercept

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the estimator based on the mode.

        :return: None
        """
        self._model = LinearRegression(fit_intercept=self.is_intercept)

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        This method allows sklearn utilities such as GridSearchCV to inspect and clone the estimator.

        :param deep: Whether to return parameters of nested estimators.
        :return: Estimator parameters.
        """
        return {
            "is_intercept": self.is_intercept,
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
                    f"Invalid parameter {key!r} for HyperOLSRegressor. "
                    f"Valid parameters are: {list(valid_params)}."
                )
            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
       Train the estimator with the given features and labels.

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
        Predict the labels for the given features using the estimator.

        :param features: The features to predict the labels for.
        :return: The predicted labels.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    @property
    def coefficient(self) -> Any:
        """
        Get the regression coefficients (weights).

        :return: The regression coefficients (weights).
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.coef_

    @property
    def intercept(self) -> Any:
        """
        Get the regression intercept (bias).

        :return: The regression intercept (bias).
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        if not self.is_intercept:
            return None
        return self._model.intercept_
