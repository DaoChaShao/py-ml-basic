#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/28 12:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   svc.py
# @Desc     :

from typing import Any, Literal, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.svm import SVC

from ...types import SVMKernelCategories
from ..base import Base


class HyperSVClassifier(Base, BaseEstimator):
    """ Support Vector Classification. """

    @validate_call
    def __init__(
            self,
            regularization_param: float = Field(1.0, gt=0),
            *,
            kernel: str | SVMKernelCategories | Literal["rbf", "linear", "poly", "sigmoid"] = SVMKernelCategories.RBF,
            degree: int = Field(3, gt=0, description="Degree of the polynomial kernel function."),
            randomness: int | None = 27,
    ) -> None:
        """
        Initialise the Support Vector Classifier.

        :param regularization_param: Regularization parameter.
        :param kernel: Kernel type.
        :param degree: Degree of the polynomial kernel function.
        :param randomness: Random state seed.
        :return: None
        """
        super().__init__()
        self.regularization_param: float = regularization_param
        self.kernel: SVMKernelCategories = SVMKernelCategories(kernel)
        self.degree: int = degree
        self.randomness: int | None = randomness

    @protectedmethod
    def _init_model(self) -> None:
        """ Initialise the Support Vector Classifier model. """
        self._model = SVC(
            C=self.regularization_param,
            kernel=self.kernel.value,
            degree=self.degree,
            gamma="scale",
            max_iter=-1,
            random_state=self.randomness,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """ Get the parameters of the Support Vector Classifier. """
        return {
            "regularization_param": self.regularization_param,
            "kernel": self.kernel,
            "degree": self.degree,
            "randomness": self.randomness,
        }

    @override
    def set_params(self, **params: Any) -> Self:
        """ Set the parameters of the Support Vector Classifier. """
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
        """
        Fit the Support Vector Classifier to the training data.

        :param features: Training features.
        :param labels: Training labels.
        :return: Fitted estimator.
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
        Predict the target of the input features.

        :param features: Input features.
        :return: Predicted target.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.predict(features)

    def predict_proba(self, features: DataFrame) -> Any:
        """
        Predict the probability of the input features belonging to each class.

        :param features: Input features.
        :return: Predicted probability.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.predict_proba(features)

    def score(self, features: DataFrame, labels: Series) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        :param features: Test features.
        :param labels: Test labels.
        :return: Mean accuracy.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return float(self._model.score(features, labels))
