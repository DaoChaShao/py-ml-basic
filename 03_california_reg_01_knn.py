#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 01:01
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_california_reg_01_knn.py
# @Desc     :

from random import randint

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    KNN,
    FeaturesNormaliser,
    grid_search_tunes,
    KNNMetrics,
    Missions,
    get_reg_labels_distribution,
    split_data,
)


def init_california_housing() -> Bunch:
    """
    Initialize the California housing dataset.

    :return: The California housing dataset.
    """
    return fetch_california_housing()


def main() -> None:
    """ Main Function """
    california = init_california_housing()
    features: ndarray = california.data
    labels: ndarray = california.target
    # pprint(features)
    # pprint(labels)

    labels: Series = Series(labels)
    get_reg_labels_distribution(labels, display=True)

    features: DataFrame = DataFrame(california.data, columns=california.feature_names)
    # print(features.head(), end="\n\n")
    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        mission=Missions.REG,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesNormaliser(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)

        # grid_params: dict = {
        #     "n_neighbors": list(range(1, 16)),
        #     "metric": [KNNMetrics.MINKOWSKI.value],
        #     "p": [1.0, 1.5, 2.0, 3.0]
        # }
        #
        # tunes = grid_search_tunes(
        #     train_features, train_labels, grid_params,
        #     mission=Missions.REG, score_strategy=RegScoreStrategies.RMSE,
        #     display=True
        # )
        #
        # params: dict = tunes.best_params
        # score: float = tunes.best_score
        # pprint(params)
        # print(f"Best Score: {score:.4f}")
        """
        {'metric': 'minkowski', 'n_neighbors': 9, 'p': 1.0}
        Best Score: -0.6032
        """

        knn = KNN(Missions.REG, n_neighbours=9, metric=KNNMetrics.MINKOWSKI, p=1.0)
        knn.train(train_features, train_labels)
        predictions = knn.predict(valid_features)
        # print(f"Predictions: {predictions}", end="\n\n")
        knn.eval_reg(valid_labels, predictions, display=True)

        row: int = randint(0, prove_features.shape[0] - 1)
        print(f"Row: {row} / {prove_features.shape[0]}")
        status, pred_label = knn.inference(
            prove_features[row:row + 1], prove_labels.iloc[row],
            mission=Missions.REG, error_thresholds=(0.6, 1.2),
            display=False
        )
        if status:
            print(
                f"{green('Correct')}! "
                f"Prediction: {pred_label}, "
                f"Label: {prove_labels.iloc[row]}."
            )
        else:
            print(
                f"{red('Incorrect')}! "
                f"Prediction: {pred_label}, "
                f"Label: {prove_labels.iloc[row]}."
            )

        # knn.save(model_name="california_knn_model")


if __name__ == "__main__":
    main()
