#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/28 12:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   svr.py
# @Desc     :   

from typing import Any, Literal, override, Self

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.svm import SVR

from ..base import Base
from ...types import SVMKernelCategories


class HyperSVRegressor(Base, BaseEstimator):
    """ Support Vector Regression. """

    @validate_call
    def __init__(
            self,
            *,
            kernel: str | SVMKernelCategories | Literal["rbf", "linear", "poly", "sigmoid"] = SVMKernelCategories.RBF,
            degree: int = Field(3, gt=0),
            regularization_param: float = Field(1.0, gt=0),
            epsilon: float = Field(0.1, ge=0),
    ) -> None:
        """
        Initialise the Support Vector Regression estimator.

        :param kernel: Kernel function type.
        :param degree: Degree of the polynomial kernel function.
        :param regularization_param: Regularization parameter.
        :param epsilon: Epsilon value for the epsilon-insensitive loss function.
        :return: None
        """
        super().__init__()
        self.kernel: SVMKernelCategories = SVMKernelCategories(kernel)
        self.degree: int = degree
        self.regularization_param: float = regularization_param
        self.epsilon: float = epsilon

    @protectedmethod
    def _init_model(self) -> None:
        """ Initialise the SVR model. """
        self._model = SVR(
            kernel=self.kernel.value,
            degree=self.degree,
            gamma="scale",
            C=self.regularization_param,
            epsilon=self.epsilon,
            max_iter=-1,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """ Get the parameters of the Support Vector Regression estimator. """
        return {
            "kernel": self.kernel,
            "degree": self.degree,
            "regularization_param": self.regularization_param,
            "epsilon": self.epsilon,
        }

    @override
    def set_params(self, **params: Any) -> Self:
        """ Set the parameters of the Support Vector Regression estimator. """
        if not params:
            return self

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for {self.__class__.__name__}."
                )

            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series | None = None) -> Self:
        """ Fit the Support Vector Regression model to the training data. """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")

        self._model.fit(features, labels)
        self._fitted = True
        return self

    @override
    def predict(self, features: DataFrame) -> Any:
        """ Predict the target of the input features. """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.predict(features)

    def score(self, features: DataFrame, labels: Series) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.

        :param features: Test features.
        :param labels: True labels for the test features.
        :return: R^2 score of the prediction.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return float(self._model.score(features, labels))
