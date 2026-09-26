#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/25 21:36
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   gbdtree.py
# @Desc     :

from typing import Any, Literal, Self, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
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
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
        Train the gradient boosting estimator.

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


class HyperGBDTree(Base, BaseEstimator):
    """ Scikit-learn compatible Gradient Boosting Decision Tree wrapper. """

    def __init__(
            self,
            mission: str | Missions | Literal["cls", "reg"],
            loss: str | TreeClsLoss | TreeRegLoss | Literal[
                "log_loss", "exponential", "squared_error", "absolute_error", "huber", "quantile",
            ],
            *,
            learning_rate: float = 0.1,
            n_estimators: int = 100,
            max_depth: int | None = 3,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_features: int | float | ForestFeaturesStrategies | Literal["sqrt", "log2"] | None = None,
            randomness: int = 27,
    ) -> None:
        """
        Initialise the Gradient Boosting Decision Tree estimator.
        :param mission: Estimator mission ('cls' or 'reg').
        :param loss: Loss function.
        :param learning_rate: Learning rate.
        :param n_estimators: Number of boosting stages.
        :param max_depth: Maximum depth of individual trees.
        :param min_samples_split: Minimum samples required to split a node.
        :param min_samples_leaf: Minimum samples required at a leaf.
        :param max_features: Number of features considered for each split.
        :param randomness: Random seed.
        :return: None
        """
        super().__init__()
        self.mission: Missions = Missions(mission)
        self.loss: TreeClsLoss | TreeRegLoss = (
            TreeClsLoss(loss) if self.mission is Missions.CLS else TreeRegLoss(loss)
        )
        self.learning_rate: float = learning_rate
        self.n_estimators: int = n_estimators
        self.max_depth: int | None = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.max_features: int | float | ForestFeaturesStrategies | None = (
            max_features if isinstance(max_features, (int, float, type(None)))
            else ForestFeaturesStrategies(max_features)
        )
        self.randomness: int = randomness

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the underlying sklearn GBDT model.

        :return: None
        """
        _estimator = GradientBoostingClassifier if self.mission is Missions.CLS else GradientBoostingRegressor
        _max_features = (
            self.max_features.value
            if isinstance(self.max_features, ForestFeaturesStrategies)
            else self.max_features
        )
        self._model = _estimator(
            loss=self.loss.value,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=_max_features,
            random_state=self.randomness,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        This method allows sklearn utilities such as GridSearchCV to inspect and clone the estimator.

        :param deep: Whether to return a deep copy of the estimator.
        :return: A GBDTResponses object of estimator parameters.
        """
        return {
            "mission": self.mission,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
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
                    f"Invalid parameter {key!r} for HyperGBDTree. "
                    f"Valid parameters are: {list(valid_params)}."
                )
            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
       Fit the GBDT model. sklearn estimators conventionally return self from fit().

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
