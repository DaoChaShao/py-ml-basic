#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/19 21:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   knn.py
# @Desc     :   

from enum import StrEnum, unique
from typing import Literal, Any

from access_modifiers import protectedmethod
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..helper import Access


@unique
class KNNModes(StrEnum):
    CLS = "cls"
    REG = "reg"


class KNN(Access):
    """ KNN class for classification and regression. """

    def __init__(self, mode: str | KNNModes | Literal["cls", "reg"], *, n_neighbours: int = 5):
        """
        Initialise the KNN class.

        :param mode: The mode of the KNN algorithm, can be "cls" for classification or "reg" for regression.
        :param n_neighbours: The number of neighbours to consider for classification or regression.
        """
        self._mode: KNNModes = KNNModes(mode)
        self._neighbors: int = n_neighbours
        self._fitted: bool = False
        self._estimator: KNeighborsClassifier | KNeighborsRegressor | None = None

        self._init_model()

    @protectedmethod
    def _init_model(self) -> None:
        """
        Initialise the KNN estimator based on the mode.

        :return: None
        """
        match self._mode:
            case KNNModes.CLS:
                self._estimator = KNeighborsClassifier(n_neighbors=self._neighbors)
            case KNNModes.REG:
                self._estimator = KNeighborsRegressor(n_neighbors=self._neighbors)
            case _:
                raise ValueError(f"Invalid mode: {self._mode!r}")

    def train(self, features: Any, labels: Any) -> None:
        """
        Train the KNN estimator with the given features and labels.

        :param features: The features to train the KNN estimator.
        :param labels: The labels to train the KNN estimator.
        :return: None
        """
        if self._estimator is None:
            raise RuntimeError("Estimator has not been initialized.")
        self._estimator.fit(features, labels)
        self._fitted = True

    def predict(self, features: Any) -> Any:
        """
        Predict the labels for the given features using the KNN estimator.

        :param features: The features to predict the labels for.
        :return: The predicted labels.
        """
        if self._estimator is None:
            raise RuntimeError("Estimator has not been initialized.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._estimator.predict(features)

    @property
    def mode(self) -> KNNModes:
        """
        Get the mode of the KNN algorithm.

        :return: The mode of the KNN algorithm.
        """
        return self._mode

    @property
    def neighbors(self) -> int:
        """
        Get the number of neighbors considered for classification or regression.

        :return: The number of neighbors considered for classification or regression.
        """
        return self._neighbors

    @property
    def knn(self) -> KNeighborsClassifier | KNeighborsRegressor | None:
        """
        Get the KNN estimator.

        :return: The KNN estimator.
        """
        return self._estimator

    def __repr__(self) -> str:
        """
        Return a string representation of the KNN object.

        :return: A string representation of the KNN object.
        """
        return f"KNN(mode={self._mode!r}, neighbors={self._neighbors!r}, knn={self._estimator!r})"


if __name__ == "__main__":
    pass
