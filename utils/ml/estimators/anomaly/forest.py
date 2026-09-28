#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/28 19:24
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   forest.py
# @Desc     :

from typing import Any, Literal, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest

from ..base import Base


class HyperIsolationForest(Base, BaseEstimator):
    """ Isolation Forest for anomaly detection. """

    @validate_call
    def __init__(
            self,
            n_estimators: int = Field(100, gt=0),
            *,
            max_samples: str | int | float | Literal["auto"] = "auto",
            contamination: str | float | Literal["auto"] = "auto",
            max_features: int | float = Field(1.0, gt=0, le=1.0),
            randomness: int | None = 27,
    ) -> None:
        """
        Initialise the Isolation Forest estimator.

        :param n_estimators: Number of isolation trees.
        :param max_features: Number or proportion of features used by each tree.
        :param randomness: Random state seed.
        :return: None
        """
        super().__init__()
        self.n_estimators: int = n_estimators
        self.max_samples: str | int | float | Literal["auto"] = max_samples
        self.contamination: str | float | Literal["auto"] = contamination
        self.max_features: int | float = max_features
        self.randomness: int = randomness

    @protectedmethod
    def _init_model(self) -> None:
        """ Initialise the Isolation Forest model. """
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.randomness,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """ Return estimator parameters. """
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "max_features": self.max_features,
        }

    @override
    def set_params(self, **params: Any) -> Self:
        """ Set estimator parameters. """
        if not params:
            return self

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for "
                    f"{self.__class__.__name__}."
                )

            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series | None = None) -> Self:
        """ Fit the Isolation Forest model. """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")

        self._model.fit(features)
        self._fitted = True
        return self

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict whether samples are normal or anomalous.

        Returns:
            1  -> normal
            -1 -> anomaly
        :param features: Test features.
        :return: Predictions.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.predict(features)

    def decision_function(self, test_features: DataFrame) -> Any:
        """
        Return anomaly scores.

        Positive values generally indicate normal samples,
        while negative values indicate anomalous samples.
        :param test_features: Test features.
        :return: Anomaly scores.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.decision_function(test_features)

    def score_samples(self, features: DataFrame) -> Any:
        """
        Return raw anomaly scores.

        Lower scores indicate more abnormal samples.
        :param features: Test features.
        :return: Raw anomaly scores.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.score_samples(features)

    def score(self, features: DataFrame) -> float:
        """
        Return the mean anomaly score of the input samples.

        :param features: Test features.
        :return: Mean anomaly score.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return float(self._model.score_samples(features).mean())
