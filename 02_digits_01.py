#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/21 22:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_digits_01.py
# @Desc     :

from pprint import pprint
from random import randint

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.datasets import load_digits
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    KNN,
    FeaturesNormaliser,
    KNNMetrics,
    KNNMissions,
    get_labels_distribution,
    split_data,
)


def init_digital_nums() -> Bunch:
    """
    Initialise the digital numbers dataset

    :return: The digital numbers dataset
    """
    return load_digits()


def main() -> None:
    """ Main Function """
    digits: Bunch = init_digital_nums()
    features: ndarray = digits.data
    labels: ndarray = digits.target
    pprint(features)
    pprint(labels)

    labels: Series = Series(digits.target)
    get_labels_distribution(labels, display=True)

    features: DataFrame = DataFrame(digits.data, columns=digits.feature_names)
    # print(features.head(), end="\n\n")
    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesNormaliser(train_features) as standardiser:
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
        {'metric': 'euclidean', 'n_neighbors': 3, 'p': 1.0}
        Best Score: 0.9840
        """

        knn = KNN(KNNMissions.CLS, n_neighbours=3, metric=KNNMetrics.EUCLIDEAN, p=1.0)
        knn.train(train_features, train_labels)
        predictions = knn.predict(valid_features)
        # print(f"Predictions: {predictions}", end="\n\n")
        knn.eval_cls(valid_labels, predictions, display=True)
        """
        ****************************************************************
        Classification Evaluation Metrics
        ----------------------------------------------------------------
        Accuracy  : 0.9926
        Precision : 0.9931
        Recall    : 0.9926
        F1_score  : 0.9927
        ****************************************************************
        """

        row: int = randint(0, prove_features.shape[0] - 1)
        print(f"Row: {row} / {prove_features.shape[0]}")
        status, pred_label = knn.inference(prove_features[row:row + 1], prove_labels.iloc[row], display=False)
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

        knn.save(model_name="digits_knn_model")


if __name__ == "__main__":
    main()
