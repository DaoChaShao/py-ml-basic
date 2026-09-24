#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/19 23:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preprocessor.py
# @Desc     :

from pathlib import Path
from pandas import option_context
from random import getstate, setstate
from random import seed as rnd_seed
from time import perf_counter
from torch import Tensor, tensor, float32
from typing import Any, Literal, Self

from access_modifiers import protectedmethod
from numpy import ndarray
from numpy import random as np_random
from numpy import unique as np_unique
from pandas import DataFrame, Series, concat, read_csv, read_excel
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.class_weight import compute_class_weight

from ..constants import WIDTH
from ..decorator import timer
from ..helper import Access
from ..highlighter import lines, stars
from .types import (
    ClsScoreStrategies,
    FileCategories,
    GridSearchTunesResponse,
    Missions,
    OneHotEncoderStrategies,
    RegScoreStrategies,
    SimpleImputerStrategies,
)


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
            file_category: str | FileCategories | Literal["csv", "excel"] = FileCategories.CSV,
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
        self._type: FileCategories = FileCategories(file_category)
        self._display: bool = display
        self._dataset: DataFrame | None = None

    @protectedmethod
    def _load_data(self):
        """
        Load data from a file based on the specified file category.

        :return: None
        """
        match self._type:
            case FileCategories.CSV:
                self._dataset = read_csv(self._path.resolve())
            case FileCategories.EXCEL:
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


@timer
def summary_dataframe(data: DataFrame, *, display: bool = True) -> None:
    """
    Print summary statistics of the data

    :param data: DataFrame containing the data
    :param display: Whether to print the summary in console.
    :return: None
    """
    with option_context(
            "display.max_columns", None,
            "display.width", 1_000,
            "display.max_colwidth", None
    ):
        print(data.head())
        lines()
        print(f"Data Amount: {len(data)}.")
        lines()
        print(data.describe())
        lines()
        print(f"Missing Values:\n{data.isnull().sum()[data.isnull().sum() > 0]}")
        lines()
        print(f"Duplicated Rows: {data.duplicated().sum()}")


class FeaturesTransformer(Access):

    def __init__(
            self,
            features: DataFrame,
            *,
            impute_num_strategy: str | SimpleImputerStrategies | Literal[
                "mean", "median", "most_frequent", "constant"
            ] = SimpleImputerStrategies.MEDIAN,
            impute_cat_strategy: str | SimpleImputerStrategies | Literal[
                "mean", "median", "most_frequent", "constant"
            ] = SimpleImputerStrategies.MOST_FREQUENT,
            one_hot_strategy: str | OneHotEncoderStrategies | Literal[
                "ignore", "error", "infrequent_if_exist"
            ] = OneHotEncoderStrategies.IGNORE,
            is_tensor: bool = False
    ) -> None:
        """
        Initialise the FeaturesTransformer pipeline for feature scaling, imputation, and encoding.

        :param features: Input DataFrame containing features.
        :param impute_num_strategy: Imputation strategy for numerical columns.
        :param impute_cat_strategy: Imputation strategy for categorical columns.
        :param one_hot_strategy: OneHotEncoder handle_unknown strategy.
        :param is_tensor: Whether to convert transformed outputs to PyTorch Tensor.
        :return: None
        """
        super().__init__()
        self._features: DataFrame = features
        self._imputer_num_strategy: SimpleImputerStrategies = SimpleImputerStrategies(impute_num_strategy)
        self._imputer_cat_strategy: SimpleImputerStrategies = SimpleImputerStrategies(impute_cat_strategy)
        self._one_hot_strategy: OneHotEncoderStrategies = OneHotEncoderStrategies(one_hot_strategy)
        self._is_tensor: bool = is_tensor
        self._transformer: ColumnTransformer | None = None
        self._transformed_data: DataFrame | Tensor | None = None

    @protectedmethod
    def _init_transformer(self) -> ColumnTransformer:
        """
        Divide the columns into numerical and categorical types and build ColumnTransformer.

        :return: Initialised ColumnTransformer instance.
        """
        # Divide the columns into numerical and categorical types
        _cols_num: list[str] = self._features.select_dtypes(
            include=["int32", "int64", "float32", "float64"],
        ).columns.tolist()
        _cols_cat: list[str] = self._features.select_dtypes(
            include=["object", "category"],
        ).columns.tolist()

        # Set a list of transformers to collect the pipelines
        _transformers: list[tuple[str, Pipeline, list[str]]] = []

        # Establish a pipe to process numerical features and handle missing values only if they exist
        if _cols_num:
            pipe_num = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy=self._imputer_num_strategy.value)),
                ("scaler", StandardScaler()),
            ])
            _transformers.append(("num", pipe_num, _cols_num))

        # Establish a pipe to process categorical features and handle missing values only if they exist
        if _cols_cat:
            pipe_cat = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy=self._imputer_cat_strategy.value)),
                ("encoder", OneHotEncoder(handle_unknown=self._one_hot_strategy.value))
            ])
            _transformers.append(("cat", pipe_cat, _cols_cat))
        # Establish a column transformer to process numerical and categorical features
        return ColumnTransformer(transformers=_transformers)

    def __enter__(self) -> Self:
        self._transformer: ColumnTransformer = self._init_transformer()
        self._transformer.fit(self._features)
        return self

    def transform(self, features: DataFrame | None = None, *, display: bool = False) -> DataFrame | Tensor:
        if self._transformer is None:
            raise RuntimeError("Transformer has not been fitted. Use within `with FeaturesTransformer(...)` context.")

        _target_features = self._features if features is None else features
        self._transformed_data = self._transformer.transform(_target_features)

        # If the processed data is a sparse matrix, convert it to a dense array
        if hasattr(self._transformed_data, "toarray"):
            self._transformed_data: ndarray = self._transformed_data.toarray()

        # Return DataFrame or Tensor
        if not self._is_tensor:
            # Rebuild the DataFrame with processed data and proper column names
            self._transformed_data: DataFrame = DataFrame(
                data=self._transformed_data, columns=self._transformer.get_feature_names_out()
            )
        else:
            # Build the torch tensor with processed data and proper column names
            # - tensor dtype is not quite suitable for PCA
            self._transformed_data: Tensor = tensor(self._transformed_data, dtype=float32)

        if display:
            print(
                f"Transformed features type is {type(self._transformed_data)}, "
                f"and its shape: {self._transformed_data.shape}"
            )
        return self._transformed_data

    def __exit__(self, *args) -> None:
        pass

    def __repr__(self):
        return (
            f"FeaturesTransformer("
            f"features={self._features!r}, "
            f"impute_num_strategy={self._imputer_num_strategy!r}, "
            f"impute_cat_strategy={self._imputer_cat_strategy!r}, "
            f"one_hot_strategy={self._one_hot_strategy!r}, "
            f"is_tensor={self._is_tensor!r}, "
            f"transformer={self._transformer!r}, "
            f"transformed_data={self._transformed_data!r})"
        )


@timer
def create_data_transformer(data: DataFrame) -> ColumnTransformer:
    """ Preprocess the data by handling missing values, scaling numerical features, and encoding categorical features.
    :param data: the DataFrame containing the selected features for training
    :return: the fitted ColumnTransformer
    """
    # Divide the columns into numerical and categorical types
    cols_num: list[str] = data.select_dtypes(include=["int32", "int64", "float32", "float64"]).columns.tolist()
    cols_cat: list[str] = data.select_dtypes(include=["object", "category"]).columns.tolist()

    # Set a list of transformers to collect the pipelines
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    # Establish a pipe to process numerical features and handle missing values only if they exist
    if cols_num:
        pipe_num = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", pipe_num, cols_num))

    # Establish a pipe to process categorical features and handle missing values only if they exist
    if cols_cat:
        pipe_cat = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", pipe_cat, cols_cat))

    # Establish a column transformer to process numerical and categorical features
    transformer: ColumnTransformer = ColumnTransformer(transformers=transformers)
    # Fit and transform the data
    transformer.fit(data)

    print(f"Preprocessed data type is {type(transformer)}")

    return transformer


@timer
def transform_data(data: DataFrame, preprocessor: ColumnTransformer, is_tensor: bool = False) -> DataFrame | Tensor:
    """ Transform the data using the provided preprocessor"""
    out = preprocessor.transform(data)

    # If the processed data is a sparse matrix, convert it to a dense array
    if hasattr(out, "toarray"):
        out: ndarray = out.toarray()

    # Return DataFrame or Tensor
    if not is_tensor:
        # Rebuild the DataFrame with processed data and proper column names
        output: DataFrame = DataFrame(data=out, columns=preprocessor.get_feature_names_out())
    else:
        # Build the torch tensor with processed data and proper column names
        # - tensor dtype is not quite suitable for PCA
        output: Tensor = tensor(out, dtype=float32)

    print(f"Preprocessed data type is {type(output)}, and its shape: {output.shape}")

    return output


@timer
def get_cls_labels_distribution(
        labels: Series,
        *,
        threshold: float = 1.5,
        display: bool = False
) -> tuple[Series, Series, float, bool]:
    """
    Check the distribution of the labels in the target variable and evaluate class balance.

    :param labels: The target label series.
    :param threshold: The Imbalance Ratio (IR) threshold to judge imbalance (default: 1.5).
    :param display: Whether to display formatted label counts, proportions, and balance verdict.
    :return: Tuple of (counts, proportions, imbalance_ratio, is_balanced)
    """
    counts: Series = labels.value_counts()
    proportions: Series = labels.value_counts(normalize=True)

    # Calculate Imbalance Rate (Max Count / Min Count)
    max_count = int(counts.max())
    min_count = int(counts.min()) if counts.min() > 0 else 1
    ir: float = round(max_count / min_count, 2)

    # Determine if the class balance is acceptable
    is_balanced: bool = ir < threshold

    if display:
        print(counts)
        lines()
        print(proportions)
        lines()
        print(f"Max Count / Min Count : {max_count} / {min_count}")
        print(f"Imbalance Ratio (IR)  : {ir:.2f}")

        # Provide an intuitive conclusion
        if ir < 1.5:
            verdict = "Balanced (Sample distribution is very balanced)"
        elif 1.5 <= ir < 3.0:
            verdict = "Mildly Imbalanced (Mild imbalance)"
        elif 3.0 <= ir < 5.0:
            verdict = "Moderately Imbalanced (Moderate imbalance, pay attention)"
        else:
            verdict = "Severely Imbalanced (Severe imbalance, resampling or weight setting required)"
        print(f"Status                : {verdict}")
    return counts, proportions, ir, is_balanced


@timer
def get_reg_labels_distribution(labels: Series, display: bool = True) -> dict:
    """
    Analyse the distribution features of continuous regression targets

    :param labels: the target variable
    :param display: Toggle for printing the distribution summary
    :return: the distribution summary
    """
    _stats: dict[str, float] = {
        "count": len(labels),
        "mean": labels.mean(),
        "std": labels.std(),
        "min": labels.min(),
        "25%": labels.quantile(0.25),
        "median": labels.median(),
        "75%": labels.quantile(0.75),
        "max": labels.max(),
        "skewness": labels.skew(),
        "kurtosis": labels.kurt(),
    }

    if display:
        for key, value in _stats.items():
            print(f"{key:<10}: {value:.4f}")
        lines()

        # Bias Situations
        if _stats["skewness"] > 1.0:
            print("Status: Highly Right-Skewed (Strongly recommend log1p transformation)")
        elif 0.5 < _stats["skewness"] <= 1.0:
            print("Status: Moderately Right-Skewed (Consider log1p transformation)")
        elif -1.0 <= _stats["skewness"] < -0.5:
            print("Status: Moderately Left-Skewed")
        elif _stats["skewness"] < -1.0:
            print("Status: Highly Left-Skewed")
        else:
            print("Status: Fairly Symmetric Distribution")
    return _stats


@timer
def encode_labels(labels: Series, *, top_n: int = 5, display: bool = False) -> tuple[Series, LabelEncoder]:
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
        name=f"{labels.name}_encoded" if labels.name else "encoded"
    )

    if display:
        # Method I
        # print(f"Label encoder classes: { {i: cat for i, cat in enumerate(encoder.classes_)} }")
        # Method II (3.12+)
        print(f"Labels encoder classes: {dict(enumerate(encoder.classes_))}")
        comparison = concat([labels.head(top_n), out.head(top_n)], axis=1, keys=["Original", "Encoded"])
        print(f"{top_n} / {len(out)} encoded labels:\n{comparison}")
    return out, encoder


@timer
def split_data(
        features: DataFrame, labels: Series,
        *,
        mission: str | Missions | Literal["cls", "reg"] = Missions.CLS,
        randomness: int = 27,
        shuffle_status: bool = True,
        display: bool = False,
) -> tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]:
    """
    Split the data into training, validation, and proving sets.

    :param features: the DataFrame of features
    :param labels: the Series of labels
    :param mission: the mission type, either "cls" for classification or "reg" for regression
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
        stratify=None if Missions(mission) == Missions.REG else labels if shuffle_status else None,
    )
    valid_features, prove_features, valid_labels, prove_labels = train_test_split(
        temp_features, temp_labels,
        test_size=0.5,
        random_state=randomness + randomness,
        shuffle=shuffle_status,
        stratify=None if Missions(mission) == Missions.REG else temp_labels if shuffle_status else None,
    )

    if display:
        _total = len(features)
        print(f"Train: {len(train_features)}/{_total} ({len(train_features) / _total:.1%}) -> {train_features.shape}")
        print(f"Valid: {len(valid_features)}/{_total} ({len(valid_features) / _total:.1%}) -> {valid_features.shape}")
        print(f"Prove: {len(prove_features)}/{_total} ({len(prove_features) / _total:.1%}) -> {prove_features.shape}")
    return train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels


class FeaturesNormaliser(Access):
    """ A class for normalising features using MinMaxScaler. """

    def __init__(self, features: DataFrame, *, min_value: int | float = 0, max_value: int | float = 1) -> None:
        """
        Initialise the Normaliser class

        :param features: The features to normalise.
        :param min_value: The minimum value of the range.
        :param max_value: The maximum value of the range.
        """
        super().__init__()
        self._features: DataFrame = features
        self._min: int | float = min_value
        self._max: int | float = max_value

        self._scaler: MinMaxScaler = MinMaxScaler(feature_range=(self._min, self._max))

    def __enter__(self) -> Self:
        """
        Fit the scaler to the features.

        :return: self
        """
        self._scaler.fit(self._features)
        return self

    def transform(self, features: DataFrame | Series | None = None) -> DataFrame:
        """
       Transform the features using the scaler.

       :param features: The features to transform.
       :return: The transformed features.
       """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for normalisation.")

        if isinstance(_features, Series):
            _features = _features.to_frame().T

        _transformed = self._scaler.transform(_features)
        return DataFrame(_transformed, columns=_features.columns, index=_features.index)

    def inverse_transform(self, features: DataFrame | Series | None = None) -> DataFrame:
        """
        Inverse transform the features using the scaler.

        :param features: The features to transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for inverse transformation.")

        if isinstance(_features, Series):
            _features = _features.to_frame().T

        _transformed = self._scaler.inverse_transform(_features)
        return DataFrame(_transformed, columns=_features.columns, index=_features.index)

    def fit_transform(self, features: DataFrame | Series | None = None) -> DataFrame:
        """
        Fit and transform the features using the scaler.

        :param features: The features to fit and transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for normalisation.")

        if isinstance(_features, Series):
            _features = _features.to_frame().T

        self._features = _features
        _transformed = self._scaler.fit_transform(_features)
        return DataFrame(_transformed, columns=_features.columns, index=_features.index)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
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


class FeaturesStandardiser(Access):
    """ A class for standardising features using StandardScaler. """

    def __init__(self, features: DataFrame) -> None:
        """
        Initialise the Standardiser class

        :param features: The features to standardise.
        """
        super().__init__()
        self._features: DataFrame = features
        self._scaler: StandardScaler = StandardScaler()

    def __enter__(self) -> Self:
        """
        Fit the scaler to the features.

        :return: self
        """
        self._scaler.fit(self._features)
        return self

    def transform(self, features: DataFrame | Series | None = None) -> DataFrame:
        """
        Transform the features using the scaler.

        :param features: The features to transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for standardisation.")

        if isinstance(_features, Series):
            _features = _features.to_frame().T

        _transformed = self._scaler.transform(_features)
        return DataFrame(_transformed, columns=_features.columns, index=_features.index)

    def inverse_transform(self, features: DataFrame | Series | None = None) -> DataFrame:
        """
        Inverse transform the features using the scaler.

        :param features: The features to transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for inverse transformation.")

        if isinstance(_features, Series):
            _features = _features.to_frame().T

        _transformed = self._scaler.inverse_transform(_features)
        return DataFrame(_transformed, columns=_features.columns, index=_features.index)

    def fit_transform(self, features: DataFrame | Series | None = None) -> DataFrame:
        """
        Fit and transform the features using the scaler.

        :param features: The features to fit and transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for standardisation.")

        if isinstance(_features, Series):
            _features = _features.to_frame().T

        self._features = _features
        _transformed = self._scaler.fit_transform(_features)
        return DataFrame(_transformed, columns=_features.columns, index=_features.index)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
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


class FeaturesRobustScaler(Access):
    """ A class for robustly scaling features using RobustScaler. """

    def __init__(self, features: DataFrame) -> None:

        """
        Initialise the RobustScaler class

        :param features: The features to robustly scale.
        """
        super().__init__()
        self._features: DataFrame = features
        self._scaler: RobustScaler = RobustScaler()

    def __enter__(self) -> Self:
        """
        Fit the scaler to the features.

        :return: self
        """
        self._scaler.fit(self._features)
        return self

    def transform(self, features: DataFrame | Series | None = None) -> DataFrame:
        """
        Transform the features using the scaler.

        :param features: The features to transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for robust scaling.")

        if isinstance(_features, Series):
            _features = _features.to_frame().T

        _transformed = self._scaler.transform(_features)
        return DataFrame(_transformed, columns=_features.columns, index=_features.index)

    def inverse_transform(self, features: DataFrame | Series | None = None) -> DataFrame:
        """
        Inverse transform the features using the scaler.

        :param features: The features to transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for inverse transformation.")

        if isinstance(_features, Series):
            _features = _features.to_frame().T

        _transformed = self._scaler.inverse_transform(_features)
        return DataFrame(_transformed, columns=_features.columns, index=_features.index)

    def fit_transform(self, features: DataFrame | Series | None = None) -> DataFrame:
        """
        Fit and transform the features using the scaler.

        :param features: The features to fit and transform.
        :return: The transformed features.
        """
        _features: Any = features if features is not None else self._features
        if _features is None:
            raise ValueError("No features provided for robust scaling.")

        if isinstance(_features, Series):
            _features = _features.to_frame().T

        self._features = _features
        _transformed = self._scaler.fit_transform(_features)
        return DataFrame(_transformed, columns=_features.columns, index=_features.index)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """ Do nothing. """
        pass

    def __repr__(self) -> str:
        """
        Return the string representation of the RobustScaler class.

        :return: The string representation of the RobustScaler class.
        """
        _shape: Any = getattr(self._features, "shape", type(self._features).__name__)
        return (
            f"FeaturesRobustScaler("
            f"features_shape={_shape}"
            f")"
        )


def euclidean_distance(x1: Any, x2: Any) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param x1: The first point.
    :param x2: The second point.
    :return: The Euclidean distance between the two points.
    """
    return sum((a - b) ** 2 for a, b in zip(x1, x2, strict=True)) ** 0.5


def manhattan_distance(x1: Any, x2: Any) -> float:
    """
    Calculate the Manhattan distance between two points.

    :param x1: The first point.
    :param x2: The second point.
    :return: The Manhattan distance between the two points.
    """
    return sum(abs(a - b) for a, b in zip(x1, x2, strict=True))


def chebyshev_distance(x1: Any, x2: Any) -> float:
    """
    Calculate the Chebyshev distance between two points.

    :param x1: The first point.
    :param x2: The second point.
    :return: The Chebyshev distance between the two points.
    """
    return max(abs(a - b) for a, b in zip(x1, x2, strict=True))


def minkowski_distance(x1: Any, x2: Any, p: float) -> float:
    """
    Calculate the Minkowski distance between two points.
    - If p = 1, it becomes the Manhattan distance.
    - If p = 2, it becomes the Euclidean distance.
    - If p = infinity, it becomes the Chebyshev distance.

    :param x1: The first point.
    :param x2: The second point.
    :param p: The order of the Minkowski distance.
    :return: The Minkowski distance between the two points.
    """
    if p < 1:
        raise ValueError("p must be greater than or equal to 1.")
    return sum(abs(a - b) ** p for a, b in zip(x1, x2, strict=True)) ** (1 / p)


def grid_search_tunes(
        train_features: DataFrame,
        train_labels: Series,
        grid_params: dict[str, list[Any]],
        *,
        mission: str | Missions | Literal["cls", "reg"] = Missions.CLS,
        cv_splits: int = 5,
        cv_shuffle: bool = True,
        randomness: int = 27,
        score_strategy: str | ClsScoreStrategies | RegScoreStrategies | Literal[
            "accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted", "roc_auc_ovr",
            "neg_root_mean_squared_error", "neg_mean_squared_error", "neg_mean_absolute_error", "r2", "neg_mean_absolute_percentage_error"
        ] = ClsScoreStrategies.F1_WEIGHTED,
        display: bool = False
) -> GridSearchTunesResponse:
    """
    Perform Grid Search CV to find optimal hyperparameters on training data.

    :param train_features: Training feature matrix.
    :param train_labels: Training label vector.
    :param grid_params: Dictionary with parameters names as keys and lists of parameter settings to try as values.
    :param mission: Type of machine learning task ("cls" for classification, "reg" for regression).
    :param cv_splits: Number of CV folds for tuning (default: 5).
    :param cv_shuffle: Whether to shuffle the training data before splitting (default: True).
    :param randomness: Random state for K-Fold splitting.
    :param score_strategy: Strategy to evaluate the performance on the cross-validated data.
    :param display: Whether to print formatted best parameters and score.
    :return: Dictionary with best_params and best_score.
    """
    is_cls: bool = Missions(mission) == Missions.CLS

    base_estimator = KNeighborsClassifier() if is_cls else KNeighborsRegressor()

    if is_cls:
        _cv = StratifiedKFold(n_splits=cv_splits, shuffle=cv_shuffle, random_state=randomness)
    else:
        _cv = KFold(n_splits=cv_splits, shuffle=cv_shuffle, random_state=randomness)

    _searcher = GridSearchCV(
        estimator=base_estimator,
        param_grid=grid_params,
        cv=_cv,
        scoring=ClsScoreStrategies(score_strategy) if is_cls else RegScoreStrategies(score_strategy),
        n_jobs=-1
    )
    _searcher.fit(train_features, train_labels)

    best_params: dict = _searcher.best_params_
    best_score: float = float(_searcher.best_score_)

    if display:
        stars()
        print(f"GridSearchCV Hyperparameter Tuning Results ({cv_splits}-Fold CV)")
        lines()
        print(f"Best Scoring Strategy   : {score_strategy}")
        print(f"Best CV Score           : {best_score:.4f}")
        print("Best Hyperparameters    :")
        for param, val in best_params.items():
            print(f"- {param:<22}: {val}")
        stars()
        print()

    return GridSearchTunesResponse(
        best_params=best_params,
        best_score=best_score
    )


@timer
def tune_optimal_reg_degree(
        reg_estimators: Any,
        *,
        train_features: DataFrame, train_labels: Series,
        valid_features: DataFrame, valid_labels: Series,
        degrees: list[int] | None = None,
        display: bool = False
) -> tuple[int, float]:
    """
    Search for the optimal polynomial degree for linear regression.

    :param reg_estimators: An instance of the Linear model wrapper.
    :param train_features: Features for training.
    :param train_labels: Labels for training.
    :param valid_features: Features for validation.
    :param valid_labels: Labels for validation.
    :param degrees: List of polynomial degrees to iterate over. Defaults to [1, 2, 3].
    :param display: Whether to print metrics for each degree.
    :return: A tuple of (best_degree, best_rmse).
    """
    _degrees: list = [1, 2, 3] if degrees is None else degrees
    _best_rmse: float = float("inf")
    _best_degree: int = _degrees[0]

    for degree in _degrees:
        _poly = PolynomialFeatures(degree=degree, include_bias=False)

        _train_poly = DataFrame(
            _poly.fit_transform(train_features),
            columns=_poly.get_feature_names_out(train_features.columns)
        )
        _valid_poly = DataFrame(
            _poly.transform(valid_features),
            columns=_poly.get_feature_names_out(valid_features.columns)
        )

        reg_estimators.train(_train_poly, train_labels)
        _predictions = reg_estimators.predict(_valid_poly)

        if display:
            print(f"Evaluating Polynomial Degree: {degree!r}.")

        _metrics = reg_estimators.eval_reg(valid_labels, _predictions, display=display)
        current_rmse = _metrics.get("rmse", float("inf"))
        if current_rmse < _best_rmse:
            _best_rmse = current_rmse
            _best_degree = degree

    if display:
        lines()
        print(f"Best Polynomial Degree: {_best_degree}, Best RMSE: {_best_rmse:.4f}")
    return _best_degree, _best_rmse


@timer
def tune_optimal_cls_degree(
        classifier: Any,
        *,
        train_features: DataFrame, train_labels: Series,
        valid_features: DataFrame, valid_labels: Series,
        degrees: list[int] | None = None,
        display: bool = False
) -> tuple[int, float]:
    """
    Search for the optimal polynomial degree for classification models.

    :param classifier: An instance of the Classification model wrapper.
    :param train_features: Features for training.
    :param train_labels: Labels for training.
    :param valid_features: Features for validation.
    :param valid_labels: Labels for validation.
    :param degrees: List of polynomial degrees to iterate over. Defaults to [1, 2, 3].
    :param display: Whether to print metrics for each degree.
    :return: A tuple of (best_degree, best_f1_score).
    """
    _degrees: list[int] = [1, 2, 3] if degrees is None else degrees
    _best_f1: float = -1.0
    _best_degree: int = _degrees[0]

    for degree in _degrees:
        _poly = PolynomialFeatures(degree=degree, include_bias=False)

        _train_poly = DataFrame(
            _poly.fit_transform(train_features),
            columns=_poly.get_feature_names_out(train_features.columns)
        )
        _valid_poly = DataFrame(
            _poly.transform(valid_features),
            columns=_poly.get_feature_names_out(valid_features.columns)
        )

        classifier.train(_train_poly, train_labels)
        _predictions = classifier.predict(_valid_poly)

        # F1-Score: higher, better
        current_f1 = f1_score(valid_labels, _predictions, average="weighted")

        if display:
            print(f"Polynomial Degree: {degree}, Weighted F1-Score: {current_f1:.4f}")

        if current_f1 > _best_f1:
            _best_f1 = current_f1
            _best_degree = degree

    if display:
        lines()
        print(f"Best Polynomial Degree: {_best_degree}, Best F1-Score: {_best_f1:.4f}")
    return _best_degree, _best_f1


@timer
def expand_polynomial_features(
        best_degree: int,
        *,
        train_features: DataFrame,
        valid_features: DataFrame,
        prove_features: DataFrame,
        display: bool = True,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Expand features into polynomial features using the specified degree.

    :param best_degree: Optimal polynomial degree.
    :param train_features: Feature DataFrame for training.
    :param valid_features: Feature DataFrame for validation.
    :param prove_features: Feature DataFrame for inference/proving.
    :param display: Whether to print shape transformations.
    :return: Tuple of transformed (train, valid, prove) DataFrames.
    """
    _poly = PolynomialFeatures(degree=best_degree, include_bias=False)

    if train_features is None or valid_features is None or prove_features is None:
        raise ValueError("train_features, valid_features, and prove_features cannot be None.")

    poly_train = DataFrame(
        _poly.fit_transform(train_features),
        columns=_poly.get_feature_names_out(train_features.columns),
        index=train_features.index
    )
    poly_valid = DataFrame(
        _poly.transform(valid_features),
        columns=_poly.get_feature_names_out(valid_features.columns),
        index=valid_features.index
    )
    poly_prove = DataFrame(
        _poly.transform(prove_features),
        columns=_poly.get_feature_names_out(prove_features.columns),
        index=prove_features.index
    )

    if display:
        print(f"Train: Original {train_features.shape} -> Poly {poly_train.shape}")
        print(f"Valid: Original {valid_features.shape} -> Poly {poly_valid.shape}")
        print(f"Prove: Original {prove_features.shape} -> Poly {poly_prove.shape}")
    return poly_train, poly_valid, poly_prove


@timer
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
