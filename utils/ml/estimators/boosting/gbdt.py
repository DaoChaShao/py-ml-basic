#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/25 21:36
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   gbdt.py
# @Desc     :

from typing import Any, Literal, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ...types import (
    ForestFeaturesStrategies,
    Missions,
    TreeClsLoss,
    TreeRegLoss,
)
from ..base import Base


class GBDTree(Base):
    """ Gradient Boosting Decision Tree Classifier/Regressor Wrapper. """

    @validate_call
    def __init__(
            self,
            mission: str | Missions | Literal["cls", "reg"],
            loss: str | TreeClsLoss | TreeRegLoss | Literal[
                "log_loss", "exponential", "squared_error", "absolute_error", "huber", "quantile"
            ],
            *,
            learning_rate: float = Field(0.1, gt=0, description="Learning rate shrinks the contribution of each tree."),
            n_estimators: int = Field(100, gt=0, description="The number of boosting stages to perform."),
            max_depth: int | None = 3,
            min_samples_split: int = Field(2, ge=2, description="Minimum samples required to split an internal node."),
            min_samples_leaf: int = Field(1, gt=0, description="Minimum samples required at a leaf node."),
            max_features: int | float | ForestFeaturesStrategies | Literal[
                "sqrt", "log2"
            ] | None = None,
            randomness: int = 27
    ) -> None:
        """
        Initialise the GBDT (Gradient Boosting Decision Tree) class.

        :param mission: The mission of the estimator ('cls' or 'reg').
        :param loss: Loss function to be optimized.
        :param learning_rate: Learning rate (shrinkage).
        :param n_estimators: Number of boosting stages.
        :param max_depth: Maximum depth of the individual regression estimators.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param min_samples_leaf: Minimum number of samples required at a leaf node.
        :param max_features: Number of features to consider when looking for the best split.
        :param randomness: Random seed for reproducibility.
        """
        super().__init__()
        self._mission: Missions = Missions(mission)
        self._loss = TreeClsLoss(loss) if self._mission is Missions.CLS else TreeRegLoss(loss)
        self._lr: float = learning_rate
        self._n_estimators: int = n_estimators
        self._max_depth: int | None = max_depth
        self._min_samples_split: int = min_samples_split
        self._min_samples_leaf: int = min_samples_leaf
        self._max_features: int | float | ForestFeaturesStrategies | None = (
            max_features if isinstance(max_features, (int, float, type(None)))
            else ForestFeaturesStrategies(max_features)
        )
        self._randomness: int = randomness

        self._init_model()

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying sklearn gradient boosting model.

        :return: None
        """
        _estimator = GradientBoostingClassifier if self._mission is Missions.CLS else GradientBoostingRegressor
        _max_features = (
            self._max_features.value
            if isinstance(self._max_features, ForestFeaturesStrategies)
            else self._max_features
        )
        self._model = _estimator(
            loss=self._loss.value,
            learning_rate=self._lr,
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf,
            max_features=_max_features,
            random_state=self._randomness,
        )

    @override
    def train(self, features: DataFrame, labels: Series) -> None:
        """
        Train the gradient boosting estimator.

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
        :return: Array of shape (n_samples, n_classes) containing predicted probabilities.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        if self._mission is not Missions.CLS:
            raise RuntimeError("Confidence probabilities are only available for classification.")
        return self._model.predict_proba(features)

    @property
    def feature_importances(self) -> Any:
        """
        Return feature importances calculated by Gini impurity / MDI.

        :return: Array of feature importances.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("Estimator has not been trained yet.")
        return self._model.feature_importances_

    def __repr__(self) -> str:
        """
        String representation of the GBDT estimator.

        :return: String representation of the GBDT estimator.
        """
        _max_features = (
            self._max_features.value
            if isinstance(self._max_features, ForestFeaturesStrategies)
            else self._max_features
        )
        return (
            f"GBDTree("
            f"mission={self._mission.value!r}, "
            f"loss={self._loss.value!r}, "
            f"n_estimators={self._n_estimators}, "
            f"learning_rate={self._lr}, "
            f"max_depth={self._max_depth}, "
            f"min_samples_split={self._min_samples_split}, "
            f"min_samples_leaf={self._min_samples_leaf}, "
            f"max_features={_max_features!r})"
        )
