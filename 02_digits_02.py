#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 00:26
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_digits_02.py
# @Desc     :

from pathlib import Path
from pprint import pprint
from random import randint

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.datasets import load_digits
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import KNN, FeaturesNormaliser, get_cls_labels_distribution, split_data


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
    get_cls_labels_distribution(labels, display=True)

    features: DataFrame = DataFrame(digits.data, columns=digits.feature_names)
    # print(features.head(), end="\n\n")
    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesNormaliser(train_features) as standardiser:
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)

        filepath: Path = Path("models/trained_at_20260922-00-25-30_digits_knn_model.pt")
        knn = KNN.load(filepath)

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


if __name__ == "__main__":
    main()
