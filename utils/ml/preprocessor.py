#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/19 23:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preprocessor.py
# @Desc     :

from enum import StrEnum, unique
from pathlib import Path
from random import getstate, setstate
from random import seed as rnd_seed
from time import perf_counter
from typing import Any, Literal, Self

from access_modifiers import protectedmethod
from numpy import ndarray
from numpy import random as np_random
from numpy import unique as np_unique
from pandas import DataFrame, Series, concat, read_csv, read_excel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from ..constants import WIDTH
from ..helper import Access


@unique
class FileCat(StrEnum):
    CSV = "csv"
    EXCEL = "excel"


class NumpySeed:
    """ Setting numpy random seed for reproducibility """

    def __init__(self, description: str, seed: int = 27, tick_tock: bool = False) -> None:
        """
        Initialise the RandomSeed class
        :param description: the description of a random seed
        :param seed: the seed value to be set
        :param tick_tock: whether to measure elapsed time
        """
        self._description: str = description
        self._seed: int = seed
        self._previous_py_seed = None
        self._previous_np_seed = None

        self._tick: bool = tick_tock
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self) -> Self:
        """ Set the random seed """
        if self._tick:
            self._start = perf_counter()

        # Save the previous random seed state
        self._previous_py_seed = getstate()
        self._previous_np_seed = np_random.get_state()

        # Set the new random seed
        rnd_seed(self._seed)
        np_random.seed(self._seed)

        print("*" * WIDTH)
        print(f"{self._description} has been set to {self._seed}.")
        print("-" * WIDTH)
        return self

    def __exit__(self, *args):
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._previous_py_seed is not None:
            setstate(self._previous_py_seed)
        if self._previous_np_seed is not None:
            np_random.set_state(self._previous_np_seed)

        # Calculate elapsed time if measuring
        if self._tick:
            self._end = perf_counter()
            self._elapsed = self._end - self._start

        print("-" * WIDTH)
        print(f"{self._description} has been restored to previous randomness.")
        if self._tick:
            elapsed_time: str = self._format_time(self._elapsed)
            print(f"{self._description} took {elapsed_time}.")
        print("*" * WIDTH)
        print()
        # Return False to propagate exceptions, True to suppress them
        return False

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format time breakdown from seconds to days, hours, minutes, and seconds
        :param seconds: time in seconds
        :return: formatted time breakdown string
        """
        if seconds < 1.0:
            return f"{seconds * 1000:.1f} ms"

        _days: int = int(seconds // 86400)
        _hours: int = int((seconds % 86400) // 3600)
        _minutes: int = int((seconds % 3600) // 60)
        _secs: float = seconds % 60

        _parts: list[str] = []
        if _days > 0:
            _parts.append(f"{_days} days")
        if _hours > 0:
            _parts.append(f"{_hours} hours")
        if _minutes > 0:
            _parts.append(f"{_minutes} minutes")
        if _secs > 0 or not _parts:
            _parts.append(f"{_secs:.2f} seconds")
        return " ".join(_parts)

    def __repr__(self) -> str:
        """ Return a string representation of the random seed """
        _base: str = f"NumpyRandomSeed(description={self._description}, seed={self._seed})"
        if self._tick and self._elapsed > 0:
            _base += f", Elapsed Time: {self._elapsed:.2f} s"
        return _base


class FileLoader(Access):

    def __init__(
            self,
            filepath: str | Path,
            *,
            file_category: str | FileCat | Literal["csv", "excel"] = FileCat.CSV,
            display: bool = True
    ) -> None:
        """
        Initialise the FileLoader class

        :param filepath: The path to the file.
        :param file_category: The category of the file (csv or excel).
        :param display: Whether to display the dataset content and summary.
        :return: None
        """
        super().__init__()
        self._path: Path = Path(filepath)
        self._type: FileCat = FileCat(file_category)
        self._display: bool = display
        self._dataset: DataFrame | None = None

    @protectedmethod
    def _load_data(self):
        """
        Load data from a file based on the specified file category.

        :return: None
        """
        match self._type:
            case FileCat.CSV:
                self._dataset = read_csv(self._path.resolve())
            case FileCat.EXCEL:
                self._dataset = read_excel(self._path.resolve())
            case _:
                raise ValueError(f"Invalid file category: {self._type}")

    def __enter__(self) -> DataFrame:
        """
        Load data from a file based on the specified file category.

        :return: The loaded dataset
        """
        self._load_data()
        if self._display:
            self._display_info()

        assert self._dataset is not None, "Dataset failed to load."
        return self._dataset

    def __exit__(self, *args) -> None:
        """
        Exit the file loader context manager.

        :param args: Exception arguments
        :return: None
        """
        pass

    @protectedmethod
    def _display_info(self) -> None:
        """
        Display information about the dataset, including its head, description, duplicated rows,
        missing values, and missing values details.

        :return: None
        """
        if self._dataset is None:
            return

        print(self._dataset.head())
        print()
        print(self._dataset.describe())
        print()
        _dup_rows: float = self._dataset.duplicated().sum()
        _miss_values: float = self._dataset.isnull().sum().sum()
        _miss_details: Series = self._dataset.isnull().sum()[self._dataset.isnull().sum() > 0]
        print(f"Duplicated Rows: {_dup_rows}")
        print(f"Missing Values: {_miss_values}")
        if not _miss_details.empty:
            print(f"Missing Values Details:\n{_miss_details}")
        else:
            print("Missing Values Details: None")

    @property
    def dataset(self) -> DataFrame | None:
        """
        Return the dataset

        :return: The dataset or None if the dataset is not loaded
        """
        return self._dataset

    def __len__(self) -> int:
        """
        Return the length of the dataset

        :return: The length of the dataset or 0 if the dataset is not loaded
        """
        return len(self._dataset) if self._dataset is not None else 0

    def __repr__(self) -> str:
        """
        Return the string representation of the FileLoader class.

        :return: The string representation of the FileLoader class.
        """
        return (
            f"FileLoader("
            f"filepath={self._path!r}, "
            f"file_category={self._type!r}, "
            f"display={self._display!r})"
        )


def check_labels_distribution(labels: Series, *, display: bool = True) -> tuple:
    """
    Check the distribution of the labels in the target variable.
    :param labels: the target variable
    :param display: Whether to display the label counts and proportions
    :return: the label counts and proportions
    """
    if display:
        print(labels.value_counts())
        print()
        print(labels.value_counts(normalize=True))

    return labels.value_counts(), labels.value_counts(normalize=True)


def encode_labels(labels: Series, *, top_n: int = 5, display: bool = True) -> tuple[Series, LabelEncoder]:
    """
    Encode the labels in the target variable.
    :param labels: the target variable
    :param top_n: the number of top labels to display
    :param display: Toggle for printing the encoded labels
    :return: the encoded labels and the label encoder
    """
    # Initialise the label encoder
    encoder: LabelEncoder = LabelEncoder()
    # Fit and transform the label encoder
    out: Series = Series(
        encoder.fit_transform(labels),
        index=labels.index,
        name=labels.name
    )

    if display:
        # Method I
        # print(f"Label encoder classes: { {i: cat for i, cat in enumerate(encoder.classes_)} }")
        # Method II
        print(f"Label encoder classes: {dict(enumerate(encoder.classes_))}")
        print(f"{top_n} / {len(out)} encoded labels:\n{concat([labels.head(top_n), out.head(top_n)], axis=1)}")
    return out, encoder


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


def split_data(
        features: DataFrame, labels: Series,
        *,
        randomness: int = 27,
        shuffle_status: bool = True,
        display: bool = True
) -> tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]:
    """
    Split the data into training, validation, and proving sets.

    :param features: the DataFrame of features
    :param labels: the Series of labels
    :param randomness: the random seed for reproducibility
    :param shuffle_status: whether to shuffle the data before splitting
    :param display: Toggle for printing the split sets
    :return: the training, validation, and proving sets
    """
    assert len(features) == len(labels), "The number of features must be equal to the number of labels."

    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features, labels,
        test_size=0.3,
        random_state=randomness,
        shuffle=shuffle_status,
        stratify=labels if shuffle_status else None,
    )
    valid_features, prove_features, valid_labels, prove_labels = train_test_split(
        temp_features, temp_labels,
        test_size=0.5,
        random_state=randomness,
        shuffle=shuffle_status,
        stratify=temp_labels if shuffle_status else None,
    )

    if display:
        print(f"Training set: {train_features.shape}, {train_labels.shape}")
        print(f"Validation set: {valid_features.shape}, {valid_labels.shape}")
        print(f"Proving set: {prove_features.shape}, {prove_labels.shape}")
    return train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels


def calc_labels_weight(labels: Any, *, display: bool = True) -> ndarray:
    """
    Compute class weight for imbalanced datasets.
    :param labels: the target variable
    :param display: Toggle for printing the class weights
    :return:
    """
    _weight: ndarray = compute_class_weight(
        class_weight="balanced",
        classes=np_unique(labels),
        y=labels,
    )

    if display:
        print(_weight)

    return _weight


if __name__ == "__main__":
    pass
