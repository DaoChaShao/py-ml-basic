#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/25 21:34
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   forest.py
# @Desc     :

from typing import Any, Literal, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ...types import (
    ForestFeaturesStrategies,
    Missions,
    TreeClsCriteria,
    TreeRegCriteria,
)
from ..base import Base


class RandomForest(Base):
    """ Random Forest Classifier/Regressor Wrapper. """

    @validate_call
    def __init__(
            self,
            mission: str | Missions | Literal["cls", "reg"],
            criterion: str | TreeClsCriteria | TreeRegCriteria | Literal[
                "gini", "entropy", "log_loss",
                "squared_error", "absolute_error", "friedman_mse", "poisson"
            ],
            *,
            n_estimators: int = Field(100, gt=0, description="The number of trees in the forest."),
            max_depth: int | None = None,
            min_samples_split: int = Field(2, ge=2, description="Minimum samples required to split an internal node."),
            min_samples_leaf: int = Field(1, gt=0, description="Minimum samples required at a leaf node."),
            max_features: int | float | ForestFeaturesStrategies | Literal[
                "sqrt", "log2"
            ] = ForestFeaturesStrategies.SQRT,
            randomness: int = 27
    ) -> None:
        """
        Initialise the RandomForest class.

        :param mission: The mission of the estimator ('cls' or 'reg').
        :param criterion: Splitting criterion.
        :param n_estimators: Number of trees in the forest.
        :param max_depth: Maximum depth of the trees.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
        :param max_features: Number of features to consider when looking for the best split.
        :param randomness: Random seed for reproducibility.
        """
        super().__init__()
        self._mission: Missions = Missions(mission)
        self.criterion: TreeClsCriteria | TreeRegCriteria = (
            TreeClsCriteria(criterion)
            if self._mission is Missions.CLS else TreeRegCriteria(criterion)
        )
        self._n_estimators: int = n_estimators
        self._max_depth: int | None = max_depth
        self._min_samples_split: int = min_samples_split
        self._min_samples_leaf: int = min_samples_leaf
        self._max_features: int | float | ForestFeaturesStrategies = (
            max_features if isinstance(max_features, (int, float)) else ForestFeaturesStrategies(max_features)
        )
        self._randomness: int = randomness

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying sklearn random forest model.

        :return: None
        """
        _estimator = RandomForestClassifier if self._mission is Missions.CLS else RandomForestRegressor
        _max_features = (
            self._max_features.value
            if isinstance(self._max_features, ForestFeaturesStrategies)
            else self._max_features
        )
        self._model = _estimator(
            n_estimators=self._n_estimators,
            criterion=self.criterion.value,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf,
            max_features=_max_features,
            random_state=self._randomness,
            n_jobs=-1
        )

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
        Train the random forest estimator.

        :param features: The features of the training data.
        :param labels: The labels of the training data.
        :return: The estimator.
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
        String representation of the RandomForest estimator.

        :return: String representation of the RandomForest estimator.
        """
        _max_features = (
            self._max_features.value
            if isinstance(self._max_features, ForestFeaturesStrategies)
            else self._max_features
        )
        return (
            f"RandomForest("
            f"mission={self._mission.value!r}, "
            f"n_estimators={self._n_estimators}, "
            f"max_depth={self._max_depth}, "
            f"min_samples_split={self._min_samples_split}, "
            f"min_samples_leaf={self._min_samples_leaf}, "
            f"max_features={_max_features!r})"
        )


class HyperRandomForest(Base, BaseEstimator):
    """ Random Forest Classifier/Regressor Wrapper. """

    @validate_call
    def __init__(
            self,
            mission: str | Missions | Literal["cls", "reg"],
            criterion: str | TreeClsCriteria | TreeRegCriteria | Literal[
                "gini", "entropy", "log_loss",
                "squared_error", "absolute_error", "friedman_mse", "poisson"
            ],
            *,
            n_estimators: int = Field(100, gt=0, description="The number of trees in the forest."),
            max_depth: int | None = None,
            min_samples_split: int = Field(2, ge=2, description="Minimum samples required to split an internal node."),
            min_samples_leaf: int = Field(1, gt=0, description="Minimum samples required at a leaf node."),
            max_features: int | float | ForestFeaturesStrategies | Literal[
                "sqrt", "log2"
            ] = ForestFeaturesStrategies.SQRT,
            randomness: int = 27
    ) -> None:
        """
        Initialise the RandomForest class.

        :param mission: The mission of the estimator ('cls' or 'reg').
        :param criterion: Splitting criterion.
        :param n_estimators: Number of trees in the forest.
        :param max_depth: Maximum depth of the trees.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
        :param max_features: Number of features to consider when looking for the best split.
        :param randomness: Random seed for reproducibility.
        """
        super().__init__()
        self.mission: Missions = Missions(mission)
        self.criterion: TreeClsCriteria | TreeRegCriteria = (
            TreeClsCriteria(criterion)
            if self.mission is Missions.CLS else TreeRegCriteria(criterion)
        )
        self.n_estimators: int = n_estimators
        self.max_depth: int | None = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.max_features: int | float | ForestFeaturesStrategies = (
            max_features if isinstance(max_features, (int, float)) else ForestFeaturesStrategies(max_features)
        )
        self.randomness: int = randomness

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying sklearn random forest model.

        :return: None
        """
        _estimator = RandomForestClassifier if self.mission is Missions.CLS else RandomForestRegressor
        _max_features = (
            self.max_features.value
            if isinstance(self.max_features, ForestFeaturesStrategies)
            else self.max_features
        )
        self._model = _estimator(
            n_estimators=self.n_estimators,
            criterion=self.criterion.value,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=_max_features,
            random_state=self.randomness,
            n_jobs=-1
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
            "criterion": self.criterion,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
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
                    f"Invalid parameter {key!r} for HyperRandomForest. "
                    f"Valid parameters are: {list(valid_params)}."
                )
            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
        Train the random forest estimator.

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
        if self.mission is not Missions.CLS:
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
