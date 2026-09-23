#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 13:50
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   elastic.py
# @Desc     :   ElasticNet (L1 + L2 Regularised) Regression Estimator Wrapper.

from typing import Any, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.linear_model import ElasticNet

from .base import Base


class ElasticNetReg(Base):

    @validate_call
    def __init__(
            self,
            *,
            alpha: float = 1.0,
            l1_ratio: float = Field(0.5, ge=0, le=1, description="l1_ratio = 1 is Lasso, l1_ratio = 0 is Ridge."),
            is_intercept: bool = True,
            epochs: int = 1_000,
            randomness: int = 27,
    ) -> None:
        """
        Initialise the ElasticNet Regression estimator.

        :param alpha: Constant that multiplies the penalty terms. Defaults to 1.0.
        :param l1_ratio: ElasticNet mixing parameter (0 <= l1_ratio <= 1).
        :param is_intercept: Whether to calculate the intercept for this model.
        :param epochs: Maximum number of iterations for coordinate descent.
        :param randomness: Seed for reproducible random state.
        :return: None
        """
        super().__init__()
        self._alpha: float = alpha
        self._l1_ratio: float = l1_ratio
        self._is_intercept: bool = is_intercept
        self._epochs: int = epochs
        self._randomness: int = randomness

        self._init_model()

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying Scikit-Learn ElasticNet estimator.

        :return: None
        """
        self._model = ElasticNet(
            alpha=self._alpha,
            l1_ratio=self._l1_ratio,
            fit_intercept=self._is_intercept,
            max_iter=self._epochs,
            random_state=self._randomness,
        )

    @override
    def train(self, features: DataFrame, labels: Series) -> None:
        """
        Train the ElasticNet Regression estimator with the given features and labels.

        :param features: The features to train the estimator.
        :param labels: The labels to train the estimator.
        :return: None
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")
        self._model.fit(features, labels)
        self._fitted = True

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict the labels for the given features using the ElasticNet estimator.

        :param features: The features to predict the labels for.
        :return: The predicted labels.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    @property
    def alpha(self) -> float:
        """
        Get the regularisation strength (alpha).

        :return: The alpha value.
        """
        return self._alpha

    @property
    def l1_ratio(self) -> float:
        """
        Get the ElasticNet mixing parameter (l1_ratio).

        :return: The l1_ratio value.
        """
        return self._l1_ratio

    @property
    def coefficient(self) -> Any:
        """
        Get the regression coefficients (weights).

        :return: The regression coefficients.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.coef_

    @property
    def intercept(self) -> Any:
        """
        Get the regression intercept (bias).

        :return: The regression intercept value or None if disabled.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        if not self._is_intercept:
            return None
        return self._model.intercept_[0] if self._model.intercept_.ndim > 0 else self._model.intercept_

    def __repr__(self) -> str:
        """
        Get the string representation of the estimator.

        :return: The string representation.
        """
        return (
            f"ElasticReg("
            f"alpha={self._alpha}, "
            f"l1_ratio={self._l1_ratio}, "
            f"intercept={self._is_intercept}, "
            f"epochs={self._epochs}, "
            f"randomness={self._randomness}, "
            f"fitted={self._fitted})"
        )
