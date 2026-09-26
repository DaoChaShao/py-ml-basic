#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/26 17:25
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   xgboost.py
# @Desc     :   

from typing import Any, Literal, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor

from ...types import (
    Missions,
    XGBClsObjectives,
    XGBRegObjectives,
)
from ..base import Base


class HyperXGBooster(Base, BaseEstimator):
    """ HyperXGBooster is a class for hyperparameter tuning of the XGBoost algorithm. """

    @validate_call
    def __init__(
            self,
            mission: str | Missions | Literal["cls", "reg"],
            objective: str | XGBClsObjectives | XGBRegObjectives | Literal[
                "binary:logistic", "binary:logitraw", "multi:softmax", "multi:softprob",
                "reg:squarederror", "reg:squaredlogerror", "reg:absoluteerror", "reg:pseudohubererror", "reg:quantileerror", "reg:huber",
            ],
            *,
            learning_rate: float = Field(0.1, le=0.1),
            n_estimators: int = 100,
            max_depth: int | None = 3,
            randomness: int = 27,
    ) -> None:
        """
        Initialise the XGBoost estimator.
        :param mission: Estimator mission ('cls' or 'reg').
        :param objective: Loss function.
        :param learning_rate: Learning rate.
        :param n_estimators: Number of boosting stages.
        :param max_depth: Maximum depth of individual trees.
        :param randomness: Random seed.
        :return: None
        """
        super().__init__()
        self.mission: Missions = Missions(mission)
        self.objective: XGBClsObjectives | XGBRegObjectives = (
            XGBClsObjectives(objective) if self.mission is Missions.CLS else XGBRegObjectives(objective)
        )
        self.learning_rate: float = learning_rate
        self.n_estimators: int = n_estimators
        self.max_depth: int | None = max_depth
        self.randomness: int = randomness

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying XGBoost model.

        :return: None
        """
        _estimator = XGBClassifier if self.mission is Missions.CLS else XGBRegressor
        self._model = _estimator(
            objective=self.objective.value,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.randomness,
            n_jobs=1,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        This method allows sklearn utilities such as GridSearchCV to inspect and clone the estimator.

        :param deep: Whether to return parameters of nested estimators.
        :return: Estimator parameters.
        """
        return {
            "mission": self.mission,
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
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
                    f"Invalid parameter {key!r} for HyperXGBooster. "
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

        if self.mission is Missions.CLS:
            self.classes_ = self._model.classes_

        self.n_features_in_ = self._model.n_features_in_
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

        if self.mission is not Missions.CLS:
            raise RuntimeError("Confidence probabilities are only available for classification.")
        return self._model.predict_proba(features)

    @property
    def feature_importances(self) -> Any:
        """
        Return feature importances.

        :return: Feature importances.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.feature_importances_
