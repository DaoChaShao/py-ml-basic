#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/20 01:03
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_iris.py
# @Desc     :

from enum import StrEnum, Enum, unique
from pandas import Series, DataFrame
from pprint import pprint
from random import uniform, randint

from pydantic import Field, validate_call
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

from utils import red, green
from utils.ml import (
    get_labels_distribution,
    FeaturesStandardiser,
    split_data,
    KNNMissions, KNNMetrics, KNN,
)


@validate_call
def init_iris(top_n: int = Field(5, gt=0, le=150), *, display: bool = False, describe: bool = False) -> Bunch:
    """
    Initialise Iris Dataset

    :param top_n: Number of samples to display
    :param describe: Whether to describe the data and metadata
    :param display: Whether to display the data and metadata
    :return: Iris Dataset
    """
    iris = load_iris()
    if display:
        pprint(iris.data[:top_n])
        pprint(iris.target[:top_n])
        print()
        print(f"Iris Dataset Features Name: {iris.feature_names}")
        print(f"Iris Dataset Labels Name: {iris.target_names}")
        print(f"Iris Dataset Features Shape: {iris.data.shape}")
        print(f"Iris Dataset Labels Shape: {iris.target.shape}")
    if describe:
        print(f"Iris Dataset Description: {iris.DESCR}")
    return iris


@unique
class Language(StrEnum):
    EN = "English"
    CN = "Chinese"


@unique
class IrisFeatures(Enum):
    SEPAL_LENGTH = ("sepal length (cm)", "花萼长度 (厘米)")
    SEPAL_WIDTH = ("sepal width (cm)", "花萼宽度 (厘米)")
    PETAL_LENGTH = ("petal length (cm)", "花瓣长度 (厘米)")
    PETAL_WIDTH = ("petal width (cm)", "花瓣宽度 (厘米)")

    def __init__(self, en_name: str, cn_name: str):
        self.EN: str = en_name
        self.CN: str = cn_name


@unique
class IrisLabels(Enum):
    SETOSA = ("setosa", "山鸢尾")
    VERSICOLOR = ("versicolor", "变色鸢尾")
    VIRGINICA = ("virginica", "维吉尼亚鸢尾")

    def __init__(self, en_name: str, cn_name: str):
        self.EN: str = en_name
        self.CN: str = cn_name


def rand_a_sample(amount: int = 1, *, display: bool = False) -> DataFrame:
    """ Random a sample """
    _statistics: dict[IrisFeatures, tuple[float, float]] = {
        IrisFeatures.SEPAL_LENGTH: (4.3, 7.9),
        IrisFeatures.SEPAL_WIDTH: (2.0, 4.4),
        IrisFeatures.PETAL_LENGTH: (1.0, 6.9),
        IrisFeatures.PETAL_WIDTH: (0.1, 2.5),
    }

    _samples = []
    for _ in range(amount):
        _sample = [round(uniform(bounds[0], bounds[1]), 2) for bounds in _statistics.values()]
        _samples.append(_sample)

    _names = [feature.EN for feature in _statistics.keys()]
    dataframe: DataFrame = DataFrame(_samples, columns=_names)

    if display:
        print(dataframe)
    return dataframe


def main() -> None:
    """ Main Function """
    iris: Bunch = init_iris()

    labels: Series = Series(iris.target)
    get_labels_distribution(labels, display=True)

    with FeaturesStandardiser(iris.data) as standardiser:
        result = standardiser.transform(iris.data)
    features: DataFrame = DataFrame(result, columns=iris.feature_names)
    # print(features.head(), end="\n\n")

    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        randomness=27, shuffle_status=True, display=True
    )

    knn: KNN = KNN(KNNMissions.CLS, n_neighbours=3, metric=KNNMetrics.MINKOWSKI, p=3.0)
    knn.train(train_features, train_labels)
    predictions = knn.predict(valid_features)
    # print(f"Predictions: {predictions}", end="\n\n")
    knn.eval_cls_metrics(valid_labels, predictions, display=True)

    row: int = randint(0, prove_features.shape[0] - 1)
    print(f"Row: {row} / {prove_features.shape[0]}")
    status, pred_label = knn.inference(prove_features.iloc[[row]], prove_labels.iloc[row], display=False)
    if status:
        print(
            f"{green('Correct')}! "
            f"Prediction: {list(IrisLabels)[pred_label].CN}, "
            f"Label: {list(IrisLabels)[prove_labels.iloc[row]].CN}."
        )
    else:
        print(
            f"{red('Incorrect')}! "
            f"Prediction: {list(IrisLabels)[pred_label].CN}, "
            f"Label: {list(IrisLabels)[prove_labels.iloc[row]].CN}."
        )

    sample = rand_a_sample(amount=1, display=True)
    pred = knn.predict(sample)[0]
    print(f"Prediction: {list(IrisLabels)[pred].CN}")
    prob = knn.confidence(sample)
    print(f"Probability: {prob}")


if __name__ == "__main__":
    main()
