#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 21:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_california_reg_03_liner_ploy.py
# @Desc     :

from pprint import pprint
from random import randint

from pandas import DataFrame, Series
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    FeaturesRobustScaler,
    LinearReg,
    Missions,
    expand_polynomial_features,
    get_reg_labels_distribution,
    split_data,
    tune_optimal_degree,
)


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

    with FeaturesRobustScaler(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)
        """
        ****************************************************************
        Regression Evaluation Metrics - degree 1
        ----------------------------------------------------------------
        R² Score  : 0.5719
        RMSE      : 0.7499
        MAE       : 0.5309
        MSE       : 0.5623
        MAPE      : 31.4934%
        ****************************************************************
        ****************************************************************
        Regression Evaluation Metrics - degree 2
        ----------------------------------------------------------------
        R² Score  : 0.6266 - higher, better (over 0.7 is good)
        RMSE      : 0.7003 - lower, better
        MAE       : 0.4653 - lower, better
        MSE       : 0.4905 - lower, better
        MAPE      : 26.3077% - lower, better
        ****************************************************************
        Best Polynomial Degree: 2, Best RMSE: 0.7003
        """

        linear = LinearReg()
        degrees: list[int] = [1, 2, 3]
        best_degree, _ = tune_optimal_degree(
            linear,
            train_features=train_features, train_labels=train_labels,
            valid_features=valid_features, valid_labels=valid_labels,
            degrees=degrees, display=False
        )

        poly_train_features, poly_valid_features, poly_prove_features = expand_polynomial_features(
            best_degree,
            train_features=train_features,
            valid_features=valid_features,
            prove_features=prove_features
        )

        linear.train(poly_train_features, train_labels)
        predictions = linear.predict(poly_valid_features)
        linear.eval_reg(valid_labels, predictions, display=True)

        row: int = randint(0, poly_prove_features.shape[0] - 1)
        print(f"Row: {row} / {poly_prove_features.shape[0]}")
        status, pred_label = linear.inference(
            poly_prove_features.iloc[row], prove_labels.iloc[row],
            mission=Missions.REG, error_thresholds=(0.70, 1.40),
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
        array([ 1.16237734e+00,  2.30301398e-01, -2.99199112e-01,  1.17262512e-01,
       -1.51290178e-02, -1.65582440e-01, -2.31926344e+00, -2.28799837e+00,
       -1.44366275e-01,  1.03474307e-01,  1.53416387e-01, -3.72259600e-02,
        1.22922043e-01, -1.51724669e-02, -1.27714528e+00, -1.25027049e+00,
        7.22777430e-02, -6.87101137e-02,  2.49002564e-02,  5.49793345e-02,
       -3.92998371e-02, -6.91907426e-01, -6.96692759e-01,  2.11734927e-02,
       -1.26039202e-02, -1.04865721e-01,  3.00616220e-02,  5.78244284e-01,
        5.78733287e-01,  1.90667267e-03,  4.99444564e-02, -4.13039539e-03,
       -1.68982438e-01, -1.66029658e-01,  2.30284225e-03,  1.71460041e-02,
        8.11414442e-02,  5.06362394e-02,  7.14640674e-05,  6.45680855e-02,
        1.84856191e-02,  8.94500362e-01,  1.54033270e+00,  5.71579919e-01])
        1.9885889286266694
        """


if __name__ == "__main__":
    main()
