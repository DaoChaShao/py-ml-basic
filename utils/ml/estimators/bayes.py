#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/26 21:42
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   bayes.py
# @Desc     :


from typing import Any, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB

from .base import Base


class NaiveBayes(Base, BaseEstimator):
    """ HyperXGBooster is a class for hyperparameter tuning of the XGBoost algorithm. """

    @validate_call
    def __init__(
            self,
            alpha: float = Field(0.1, description="Laplace or Lidstone smoothing parameter."),
            *,
            force_alpha: bool = Field(True, description="Whether to use Laplace smoothing."),
    ) -> None:
        """
        Initialise the NaiveBayes estimator.
        :param alpha: Learning rate.
        :param force_alpha: Number of boosting stages.
        :return: None
        """
        super().__init__()
        self.alpha: float = alpha
        self.force_alpha: bool = force_alpha

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying XGBoost model.

        :return: None
        """
        self._model = MultinomialNB(
            alpha=self.alpha,
            force_alpha=self.force_alpha,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        This method allows sklearn utilities such as GridSearchCV to inspect and clone the estimator.

        :param deep: Whether to return parameters of nested estimators.
        :return: Estimator parameters.
        """
        return {
            "alpha": self.alpha,
            "force_alpha": self.force_alpha,
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
                    f"Invalid parameter {key!r} for NaiveBayes. "
                    f"Valid parameters are: {list(valid_params)}."
                )
            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
       Fit the XGBoost model. sklearn estimators conventionally return self from fit().

       :param features: The features of the training data.
       :param labels: The labels of the training data.
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

    def confidence(self, features: DataFrame) -> Any:
        """
        Predict class probabilities for classification tasks.

        :param features: The features of the input data.
        :return: Predicted class probabilities.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.predict_proba(features)
