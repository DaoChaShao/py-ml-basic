#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 22:54
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   04_diabetes_reg_01_linear.py
# @Desc     :

from pprint import pprint
from random import randint

from pandas import DataFrame, Series
from sklearn.datasets import load_diabetes
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    FeaturesRobustScaler,
    Missions,
    expand_polynomial_features,
    get_reg_labels_distribution,
    split_data,
    tune_optimal_reg_degree,
)
from utils.ml.estimators import OLSReg


def init_diabetes() -> Bunch:
    """
    Initialise the diabetes dataset.

    :return: The diabetes dataset.
    """
    return load_diabetes()


def main() -> None:
    """ Main Function """
    diabetes = init_diabetes()

    labels: Series = Series(diabetes.target)
    get_reg_labels_distribution(labels, display=True)

    features: DataFrame = DataFrame(diabetes.data, columns=diabetes.feature_names)
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

        linear = OLSReg()
        degrees: list[int] = [1, 2, 3]
        best_degree, best_rmse = tune_optimal_reg_degree(
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
        metrics = linear.eval_reg(valid_labels, predictions, display=True)
        """
        ****************************************************************
        Regression Evaluation Metrics
        ----------------------------------------------------------------
        R² Score  : 0.3540
        RMSE      : 51.4414
        MAE       : 42.4228
        MSE       : 2646.2175
        MAPE      : 35.5421%
        ****************************************************************
        """

        curr_rmse: float = metrics.get("rmse", best_rmse)

        row: int = randint(0, poly_prove_features.shape[0] - 1)
        print(f"Row: {row} / {poly_prove_features.shape[0]}")
        status, pred_label = linear.inference(
            poly_prove_features.iloc[row:row + 1], prove_labels.iloc[row],
            mission=Missions.REG, reg_error_thresholds=(curr_rmse, curr_rmse * 2),
            display=False
        )
        if status:
            print(
                f"{green('Correct')}! "
                f"Prediction: {pred_label}, "
                f"Label: {prove_labels.iloc[row]}."
            )
            print()
        else:
            print(
                f"{red('Incorrect')}! "
                f"Prediction: {pred_label}, "
                f"Label: {prove_labels.iloc[row]}."
            )
            print()

        print("Coefficients:")
        pprint(linear.coefficient)
        print(f"Intercept: {linear.intercept}")
        """
        Coefficients:
        array([  1.29605856, -22.07030134,  39.21426325,  20.88183981,
               -34.69653007,  17.9623933 ,   6.6868702 ,  19.84714902,
                39.67536732,   5.45957327])
        Intercept: 152.31351288142994
        """


if __name__ == "__main__":
    main()
