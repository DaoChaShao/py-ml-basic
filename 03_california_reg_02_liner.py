#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 19:11
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_california_reg_02_liner.py
# @Desc     :

from pprint import pprint
from random import randint

from pandas import DataFrame, Series
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    FeaturesNormaliser,
    Missions,
    get_reg_labels_distribution,
    split_data,
)
from utils.ml.estimators import OLSReg


def init_california_housing() -> Bunch:
    """
    Initialise the California housing dataset.

    :return: The California housing dataset.
    """
    return fetch_california_housing()


def main() -> None:
    """ Main Function """
    california = init_california_housing()

    labels: Series = Series(california.target)
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
        """
        ****************************************************************
        Regression Evaluation Metrics - FeaturesNormaliser
        ----------------------------------------------------------------
        R² Score  : 0.5719
        RMSE      : 0.7499
        MAE       : 0.5309
        MSE       : 0.5623
        MAPE      : 31.4934%
        ****************************************************************
        """

        linear = OLSReg()
        linear.train(train_features, train_labels)
        predictions = linear.predict(valid_features)
        linear.eval_reg(valid_labels, predictions, display=True)

        row: int = randint(0, prove_features.shape[0] - 1)
        print(f"Row: {row} / {prove_features.shape[0]}")
        status, pred_label = linear.inference(
            prove_features[row:row + 1], prove_labels.iloc[row],
            mission=Missions.REG, error_thresholds=(0.75, 1.50),
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

        pprint(linear.coefficient)
        print(linear.intercept)
        """
        array([  6.47239289,   0.50032988, -17.10649225,  20.16492109,
         0.02308527,  -4.13769135,  -3.95736822,  -4.32796381])
        3.5884511215585335
        """


if __name__ == "__main__":
    main()
