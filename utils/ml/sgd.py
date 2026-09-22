#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 23:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   sgd.py
# @Desc     :   Stochastic Gradient Descent Regression Estimator Wrapper.

from typing import Any, Literal, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from sklearn.linear_model import SGDRegressor

from .base import Base
from .types import AlphaCategories, RegLosses


class SGDReg(Base):

    def __init__(
            self,
            *,
            loss: str | RegLosses | Literal[
                "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
            ] = RegLosses.SQUARED_ERROR,
            alpha: float = 0.0001,
            is_intercept: bool = True,
            epochs: int = 1_000,
            randomness: int = 27,
            lr_category: str | AlphaCategories | Literal[
                "invscaling", "constant", "optimal", "adaptive"
            ] = AlphaCategories.INVSCALING,
    ) -> None:
        """
        Initialise the SGD Regression estimator.

        :param loss: The loss function to be used.
        :param alpha: The regularisation strength.
        :param is_intercept: Whether to calculate the intercept for this model.
        :param epochs: Maximum number of passes over the training data.
        :param randomness: Seed for reproducible random state.
        :param lr_category: The learning rate category.
        :return: None
        """
        super().__init__()
        self._loss: RegLosses = RegLosses(loss)
        self._alpha: float = alpha
        self._is_intercept: bool = is_intercept
        self._epochs: int = epochs
        self._randomness: int = randomness
        self._lr_category: AlphaCategories = AlphaCategories(lr_category)

        self._init_model()

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the estimator based on the specified parameters.

        :return: None
        """
        self._model = SGDRegressor(
            loss=self._loss.value,
            alpha=self._alpha,
            fit_intercept=self._is_intercept,
            max_iter=self._epochs,
            random_state=self._randomness,
            learning_rate=self._lr_category.value
        )

    @override
    def train(self, features: DataFrame, labels: Series) -> None:
        """
        Train the SGD Regression estimator with the given features and labels.

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
        Predict the labels for the given features using the SGD Regression estimator.

        :param features: The features to predict the labels for.
        :return: The predicted labels.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialized.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    @property
    def loss(self) -> RegLosses:
        """
        Get the loss function.

        :return: The loss function.
        """
        return self._loss

    @property
    def alpha(self) -> float:
        """
        Get the regularisation strength.

        :return: The regularisation strength.
        """
        return self._alpha

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
        return self._model.intercept_[0] if self._model.intercept_.ndim > 0 else self._model.intercept_

    @property
    def lr_category(self) -> AlphaCategories:
        """
        Get the learning rate category.

        :return: The learning rate category.
        """
        return self._lr_category

    def __repr__(self) -> str:
        """
        Get the string representation of the estimator.

        :return: The string representation of the estimator.
        """
        return (
            f"SGDReg("
            f"loss={self._loss.value}, "
            f"alpha={self._alpha}, "
            f"intercept={self._is_intercept}, "
            f"max_iter={self._epochs}, "
            f"randomness={self._randomness}, "
            f"lr_category={self._lr_category.value}, "
            f"fitted={self._fitted}"
        )
