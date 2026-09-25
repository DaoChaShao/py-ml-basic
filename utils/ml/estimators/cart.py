#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 23:37
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cart.py
# @Desc     :

from typing import Any, Literal, override

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..types import (
    Missions,
    TreeClsCriteria,
    TreeRegCriteria,
    TreeSplitters,
)
from .base import Base


class DecisionTree(Base):
    """ CART Classifier/Regressor Wrapper. """

    @validate_call
    def __init__(
            self,
            mission: str | Missions | Literal["cls", "reg"],
            criterion: str | TreeClsCriteria | TreeRegCriteria | Literal[
                "gini", "entropy", "log_loss",
                "squared_error", "friedman_mse", "absolute_error", "poisson"
            ],
            *,
            splitter: str | TreeSplitters | Literal["best", "random"] = TreeSplitters.BEST,
            max_depth: int | None = None,
            min_samples_split: int = Field(2, ge=2, description="Minimum samples required to split an internal node."),
            min_samples_leaf: int = Field(1, gt=0, description="Minimum samples required at a leaf node."),
            max_features: int | float | None = None,
            randomness: int = 27
    ) -> None:
        """
        Initialise the CARTree class.

        :param mission: The mission of the estimator ('cls' or 'reg').
        :param criterion: Splitting criterion. Defaults to 'gini' for CLS and 'squared_error' for REG.
        :param splitter: Strategy used to choose the split at each node.
        :param max_depth: Maximum depth of the tree.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
        :param max_features: Number of features to consider when looking for the best split.
        :param randomness: Random seed for reproducibility.
        :return: None
        """
        super().__init__()
        self._mission: Missions = Missions(mission)
        self._criterion: TreeClsCriteria | TreeRegCriteria = (
            TreeClsCriteria(criterion) if self._mission is Missions.CLS else TreeRegCriteria(criterion)
        )
        self._splitter: TreeSplitters = TreeSplitters(splitter)
        self._max_depth: int | None = max_depth
        self._min_samples_split: int = min_samples_split
        self._min_samples_leaf: int = min_samples_leaf
        self._max_features: int | float | None = max_features
        self._randomness: int = randomness

        self._init_model()

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the model

        :return: None
        """
        _estimator = DecisionTreeClassifier if self._mission is Missions.CLS else DecisionTreeRegressor
        self._model = _estimator(
            criterion=self._criterion.value,
            splitter=self._splitter.value,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf,
            max_features=self._max_features,
            random_state=self._randomness,
        )

    @override
    def train(self, features: DataFrame, labels: Series) -> None:
        """
        Train the estimator.

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
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    def confidence(self, features: DataFrame) -> Any:
        """
        Predict class probabilities for classification tasks.

        :param features: The features of the input data.
        :return: Array of shape (n_samples, n_classes) containing predicted probabilities.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        if self._mission is not Missions.CLS:
            raise RuntimeError("Confidence probabilities are only available for classification.")
        return self._model.predict_proba(features)

    @property
    def feature_importances(self) -> Any:
        """
        Return feature importances calculated by Gini impurity / MDI.

        :return: Array of feature importances.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.feature_importances_

    @property
    def max_depth(self) -> int | None:
        """
        Return the maximum depth of the tree.

        :return: The maximum depth of the tree.
        """
        return self._max_depth

    @property
    def min_samples_split(self) -> int:
        """
        Return the minimum number of samples required to split an internal node.

        :return: The minimum number of samples required to split an internal node.
        """
        return self._min_samples_split

    @property
    def min_samples_leaf(self) -> int:
        """
       Return the minimum number of samples required to be at a leaf node.

       :return: The minimum number of samples required to be at a leaf node.
       """
        return self._min_samples_leaf

    @property
    def max_features(self) -> int | float | None:
        """
        Return the number of features considered when looking for the best split.

        :return: The number of features considered when looking for the best split.
        """
        return self._max_features

    def __repr__(self) -> str:
        """
        String representation of the CART estimator.

        :return: String representation of the CART estimator.
        """
        return (
            f"DecisionTree("
            f"mission={self._mission.value!r}, "
            f"criterion={self._criterion.value!r}, "
            f"splitter={self._splitter.value!r}, "
            f"max_depth={self._max_depth}, "
            f"min_samples_split={self._min_samples_split}, "
            f"min_samples_leaf={self._min_samples_leaf})"
        )
