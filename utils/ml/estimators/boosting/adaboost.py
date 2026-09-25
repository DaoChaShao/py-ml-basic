#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/25 21:54
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   adaboost.py
# @Desc     :

from typing import Any, Literal, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ...types import AdaBoostRegLoss, Missions
from ..base import Base


class AdaBoost(Base):
    """ AdaBoost Classifier/Regressor Wrapper. """

    @validate_call
    def __init__(
            self,
            mission: str | Missions | Literal["cls", "reg"],
            *,
            n_estimators: int = Field(50, gt=0, description="The maximum number of estimators."),
            learning_rate: float = Field(1.0, gt=0, description="Weight applied to each classifier."),
            loss: str | AdaBoostRegLoss | Literal["linear", "square", "exponential"] = AdaBoostRegLoss.LINEAR,
            max_depth: int = Field(1, gt=0, description="Maximum depth of the base decision tree."),
            randomness: int = 27
    ) -> None:
        """
        Initialise the AdaBoost classifier/regressor.

        :param mission: The mission of the estimator ('cls' or 'reg').
        :param n_estimators: Maximum number of boosting stages.
        :param learning_rate: Weight applied to each boosting stage.
        :param loss: Loss function for regression tasks.
        :param max_depth: Maximum depth of the base decision tree.
        :param randomness: Random seed for reproducibility.
        :return: None
        """
        super().__init__()
        self._mission: Missions = Missions(mission)
        self._n_estimators: int = n_estimators
        self._lr: float = learning_rate
        self._loss: AdaBoostRegLoss = AdaBoostRegLoss(loss)
        self._max_depth: int = max_depth
        self._randomness: int = randomness

        self._init_model()

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying sklearn AdaBoost model.

        :return: None
        """
        _base_estimator = (
            DecisionTreeClassifier(max_depth=self._max_depth)
            if self._mission is Missions.CLS
            else DecisionTreeRegressor(max_depth=self._max_depth)
        )

        match self._mission:
            case Missions.CLS:
                self._model = AdaBoostClassifier(
                    estimator=_base_estimator,
                    n_estimators=self._n_estimators,
                    learning_rate=self._lr,
                    random_state=self._randomness,
                )
            case Missions.REG:
                self._model = AdaBoostRegressor(
                    estimator=_base_estimator,
                    n_estimators=self._n_estimators,
                    learning_rate=self._lr,
                    loss=self._loss.value,
                    random_state=self._randomness,
                )
            case _:
                raise ValueError(f"Invalid mission: {self._mission}")

    @override
    def train(self, features: DataFrame, labels: Series) -> None:
        """
        Train the AdaBoost estimator.

        :param features: The features of the training data.
        :param labels: The labels of the training data.
        :return: None
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")

        self._model.fit(features, labels)
        self._fitted = True

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict the labels of the input features.

        :param features: The features of the input data.
        :return: The predicted labels.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")

        return self._model.predict(features)

    def confidence(self, features: DataFrame) -> Any:
        """
        Predict class probabilities for classification tasks.

        :param features: The features of the input data.
        :return: Array of shape (n_samples, n_classes) containing
            predicted probabilities.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")

        if self._mission is not Missions.CLS:
            raise RuntimeError(
                "Confidence probabilities are only available for classification."
            )

        return self._model.predict_proba(features)

    @property
    def feature_importances(self) -> Any:
        """
        Return feature importances calculated from the AdaBoost ensemble.

        :return: Array of feature importances.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")

        return self._model.feature_importances_

    def __repr__(self) -> str:
        """
        String representation of the AdaBoost estimator.

        :return: String representation of the AdaBoost estimator.
        """
        match self._mission:
            case Missions.CLS:
                return (
                    f"AdaBoost("
                    f"mission={self._mission.value!r}, "
                    f"n_estimators={self._n_estimators}, "
                    f"learning_rate={self._lr}, "
                    f"max_depth={self._max_depth})"
                )
            case Missions.REG:
                return (
                    f"AdaBoost("
                    f"mission={self._mission.value!r}, "
                    f"n_estimators={self._n_estimators}, "
                    f"learning_rate={self._lr}, "
                    f"loss={self._loss.value!r}, "
                    f"max_depth={self._max_depth})"
                )
            case _:
                raise ValueError(f"Invalid mission: {self._mission}")
