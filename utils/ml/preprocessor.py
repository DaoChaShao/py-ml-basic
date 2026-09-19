#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/19 23:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preprocessor.py
# @Desc     :

from typing import Any, Self

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..helper import Access


class Normaliser(Access):
    """ A class for normalising features using MinMaxScaler. """

    def __init__(self, features: Any, *, min_value: int | float = 0, max_value: int | float = 1) -> None:
        """
        Initialise the Normaliser class

        :param features: The features to normalise.
        :param min_value: The minimum value of the range.
        :param max_value: The maximum value of the range.
        """
        super().__init__()
        self._features: Any = features
        self._min: int | float = min_value
        self._max: int | float = max_value

        self._scaler: MinMaxScaler = MinMaxScaler(feature_range=(self._min, self._max))

    def __enter__(self) -> Self:
        """
        Fit the scaler to the features.

        :return: self
        """
        if self._features is not None:
            self._scaler.fit(self._features)
        return self

    def transform(self, features: Any = None) -> Any:
        """
       Transform the features using the scaler.

       :param features: The features to transform.
       :return: The transformed features.
       """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for normalisation.")
        return self._scaler.transform(_features)

    def inverse_transform(self, features: Any = None) -> Any:
        """
        Transform the features using the scaler.

        :param features: The features to transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for inverse transformation.")
        return self._scaler.inverse_transform(_features)

    def fit_transform(self, features: Any = None) -> Any:
        """
        Fit and transform the features using the scaler.

        :param features: The features to fit and transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for normalisation.")
        return self._scaler.fit_transform(_features)

    def __exit__(self, exc_type, exc_value, traceback):
        """ Do nothing. """
        pass

    def __repr__(self) -> str:
        """
        Return the string representation of the Normaliser class.

        :return: The string representation of the Normaliser class.
        """
        _shape: Any = getattr(self._features, "shape", type(self._features).__name__)
        return (
            f"Normaliser("
            f"features_shape={_shape}, "
            f"min_value={self._min!r}, "
            f"max_value={self._max!r}"
            f")"
        )


class Standardiser(Access):
    """ A class for standardising features using StandardScaler. """

    def __init__(self, features: Any = None) -> None:
        """
        Initialise the Standardiser class

        :param features: The features to standardise.
        """
        super().__init__()
        self._features: Any = features
        self._scaler: StandardScaler = StandardScaler()

    def __enter__(self) -> Self:
        """
        Fit the scaler to the features.

        :return: self
        """
        if self._features is not None:
            self._scaler.fit(self._features)
        return self

    def transform(self, features: Any = None) -> Any:
        """
        Transform the features using the scaler.

        :param features: The features to transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for standardisation.")
        return self._scaler.transform(_features)

    def inverse_transform(self, features: Any = None) -> Any:
        """
        Transform the features using the scaler.

        :param features: The features to transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for inverse transformation.")
        return self._scaler.inverse_transform(_features)

    def fit_transform(self, features: Any = None) -> Any:
        """
        Fit and transform the features using the scaler.

        :param features: The features to fit and transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for standardisation.")
        return self._scaler.fit_transform(_features)

    def __exit__(self, exc_type, exc_value, traceback):
        """ Do nothing. """
        pass

    def __repr__(self) -> str:
        """
        Return the string representation of the Standardiser class.

        :return: The string representation of the Standardiser class.
        """
        _shape: Any = getattr(self._features, "shape", type(self._features).__name__)
        return (
            f"Standardiser("
            f"features_shape={_shape}"
            f")"
        )


if __name__ == "__main__":
    pass
