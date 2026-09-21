#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/20 01:03
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_iris.py
# @Desc     :

from pprint import pprint
from random import randint, uniform

from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    KNN,
    FeaturesStandardiser,
    IrisFeatures,
    IrisLabels,
    KNNMetrics,
    KNNMissions,
    get_labels_distribution,
    split_data,
    grid_search_tunes,
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

    _names = [feature.EN for feature in _statistics]
    dataframe: DataFrame = DataFrame(_samples, columns=_names)

    if display:
        print(dataframe)
    return dataframe


def main() -> None:
    """ Main Function """
    iris: Bunch = init_iris()

    labels: Series = Series(iris.target)
    get_labels_distribution(labels, display=True)

    features: DataFrame = DataFrame(iris.data, columns=iris.feature_names)
    # print(features.head(), end="\n\n")
    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesStandardiser(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)

        # grid_params: dict = {
        #     "n_neighbors": list(range(1, 16)),
        #     "metric": [
        #         KNNMetrics.EUCLIDEAN.value,
        #         KNNMetrics.MANHATTAN.value,
        #         KNNMetrics.CHEBYSHEV.value,
        #         KNNMetrics.MINKOWSKI.value,
        #     ],
        #     "p": [1.0, 2.0, 3.0, 4.0]
        # }
        #
        # tunes = grid_search_tunes(train_features, train_labels, grid_params, display=True)
        #
        # params: dict = tunes.best_params
        # score: float = tunes.best_score
        # pprint(params)
        # print(f"Best Score: {score:.4f}")
        """
        {'metric': <KNNMetrics.MINKOWSKI: 'minkowski'>, 'n_neighbors': 6, 'p': 4.0}
        Best Score: 0.9809
        """

        knn = KNN(KNNMissions.CLS, n_neighbours=6, metric=KNNMetrics.MINKOWSKI, p=4.0)
        knn.train(train_features, train_labels)
        predictions = knn.predict(valid_features)
        # print(f"Predictions: {predictions}", end="\n\n")
        knn.eval_cls(valid_labels, predictions, display=True)

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
        sample_features = standardiser.transform(sample)
        pred = knn.predict(sample_features)[0]
        print(f"Prediction: {list(IrisLabels)[pred].CN}")
        probs = knn.confidence(sample_features)[0]
        for label, prob in zip(IrisLabels, probs, strict=True):
            print(f"{label.CN:<6}: {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
