#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 19:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   liner.py
# @Desc     :

from typing import Any, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression

from .base import Base


class LinearReg(Base):

    def __init__(self, is_intercept: bool = True) -> None:
        """
        Initialise the Linear Regression estimator.

        :param is_intercept: Whether to include an intercept term in the model. Default is True.
        :return: None
        """
        super().__init__()
        self._is_intercept: bool = is_intercept

        self._init_model()

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the estimator based on the mode.

        :return: None
        """
        self._model = LinearRegression(fit_intercept=self._is_intercept)

    @override
    def train(self, features: DataFrame, labels: Series) -> None:
        """
       Train the estimator with the given features and labels.

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
    def intercept(self) -> float:
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
