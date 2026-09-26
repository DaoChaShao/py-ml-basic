#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 01:33
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   ridge.py
# @Desc     :

from typing import Any, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from utils.ml.estimators.base import Base


class RidgeRegressor(Base):
    """ Ridge Regression Estimator Wrapper. """

    @validate_call
    def __init__(
            self,
            *,
            penalty_strength: float = Field(1.0, gt=0, description="Bigger strength, stronger regularisation."),
            is_intercept: bool = True,
            epochs: int = Field(1_000, gt=0, description="Maximum number of iterations for conjugate gradient solver."),
            randomness: int = 27,
    ) -> None:
        """
        Initialise the Ridge Regression estimator.

        :param penalty_strength: Constant that multiplies the L2 term. Defaults to 1.0.
        :param is_intercept: Whether to calculate the intercept for this model.
        :param epochs: Maximum number of iterations for conjugate gradient solver.
        :param randomness: Seed for reproducible random state.
        :return: None
        """
        super().__init__()
        self._strength: float = penalty_strength
        self._is_intercept: bool = is_intercept
        self._epochs: int = epochs
        self._randomness: int = randomness

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying Scikit-Learn Ridge estimator.

        :return: None
        """
        self._model = Ridge(
            alpha=self._strength,
            fit_intercept=self._is_intercept,
            max_iter=self._epochs,
            random_state=self._randomness,
        )

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
        Train the Ridge Regression estimator with the given features and labels.

        :param features: The features to train the estimator.
        :param labels: The labels to train the estimator.
        :return: The estimator
        """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")

        self._model.fit(features, labels)
        self._fitted = True
        return self

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict the labels for the given features using the Ridge estimator.

        :param features: The features to predict the labels for.
        :return: The predicted labels.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    @property
    def penalty_strength(self) -> float:
        """
        Get the regularisation strength (alpha).

        :return: The alpha value.
        """
        return self._strength

    @property
    def coefficient(self) -> Any:
        """
        Get the regression coefficients (weights).

        :return: The regression coefficients.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.coef_

    @property
    def intercept(self) -> Any:
        """
        Get the regression intercept (bias).

        :return: The regression intercept value or None if disabled.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
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
            f"RidgeReg("
            f"penalty_strength={self._strength}, "
            f"intercept={self._is_intercept}, "
            f"epochs={self._epochs}, "
            f"randomness={self._randomness}, "
            f"fitted={self._fitted})"
        )


class HyperRidgeRegressor(Base, BaseEstimator):
    """ Ridge Regression Estimator Wrapper. """

    @validate_call
    def __init__(
            self,
            *,
            penalty_strength: float = Field(1.0, gt=0, description="Bigger strength, stronger regularisation."),
            is_intercept: bool = True,
            epochs: int = Field(1_000, gt=0, description="Maximum number of iterations for conjugate gradient solver."),
            randomness: int = 27,
    ) -> None:
        """
        Initialise the Ridge Regression estimator.

        :param penalty_strength: Constant that multiplies the L2 term. Defaults to 1.0.
        :param is_intercept: Whether to calculate the intercept for this model.
        :param epochs: Maximum number of iterations for conjugate gradient solver.
        :param randomness: Seed for reproducible random state.
        :return: None
        """
        super().__init__()
        self.strength: float = penalty_strength
        self.is_intercept: bool = is_intercept
        self.epochs: int = epochs
        self.randomness: int = randomness

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying Scikit-Learn Ridge estimator.

        :return: None
        """
        self._model = Ridge(
            alpha=self.strength,
            fit_intercept=self.is_intercept,
            max_iter=self.epochs,
            random_state=self.randomness,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        This method allows sklearn utilities such as GridSearchCV to inspect and clone the estimator.

        :param deep: Whether to return parameters of nested estimators.
        :return: Estimator parameters.
        """
        return {
            "penalty_strength": self.strength,
            "is_intercept": self.is_intercept,
            "epochs": self.epochs,
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
                    f"Invalid parameter {key!r} for HyperRidgeRegressor. "
                    f"Valid parameters are: {list(valid_params)}."
                )
            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
        Train the Ridge Regression estimator with the given features and labels.

        :param features: The features to train the estimator.
        :param labels: The labels to train the estimator.
        :return: The estimator
        """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")

        self._model.fit(features, labels)
        self._fitted = True
        return self

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict the labels for the given features using the Ridge estimator.

        :param features: The features to predict the labels for.
        :return: The predicted labels.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    @property
    def penalty_strength(self) -> float:
        """
        Get the regularisation strength (alpha).

        :return: The alpha value.
        """
        return self.strength

    @property
    def coefficient(self) -> Any:
        """
        Get the regression coefficients (weights).

        :return: The regression coefficients.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.coef_

    @property
    def intercept(self) -> Any:
        """
        Get the regression intercept (bias).

        :return: The regression intercept value or None if disabled.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        if not self.is_intercept:
            return None
        return self._model.intercept_[0] if self._model.intercept_.ndim > 0 else self._model.intercept_
