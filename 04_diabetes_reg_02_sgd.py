#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 23:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   04_diabetes_reg_02_sgd.py
# @Desc     :

from pprint import pprint
from random import randint

from pandas import DataFrame, Series
from sklearn.datasets import load_diabetes
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    AlphaCategories,
    FeaturesRobustScaler,
    Missions,
    RegLosses,
    SGDReg,
    expand_polynomial_features,
    get_reg_labels_distribution,
    split_data,
    tune_optimal_degree,
)


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

        sgd = SGDReg(loss=RegLosses.SQUARED_ERROR, lr_category=AlphaCategories.INVSCALING)
        degrees: list[int] = [1, 2, 3]
        best_degree, best_rmse = tune_optimal_degree(
            sgd,
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

        sgd.train(poly_train_features, train_labels)
        predictions = sgd.predict(poly_valid_features)
        metrics = sgd.eval_reg(valid_labels, predictions, display=True)
        """
        ****************************************************************
        Regression Evaluation Metrics - Linear Regression
        ----------------------------------------------------------------
        R² Score  : 0.3540
        RMSE      : 51.4414
        MAE       : 42.4228
        MSE       : 2646.2175
        MAPE      : 35.5421%
        ****************************************************************
        ****************************************************************
        Regression Evaluation Metrics - SGD Regression
        ----------------------------------------------------------------
        R² Score  : 0.3523
        RMSE      : 51.5088
        MAE       : 42.5478
        MSE       : 2653.1516
        MAPE      : 35.3433%
        ****************************************************************
        """

        curr_rmse: float = metrics.get("rmse", best_rmse)

        row: int = randint(0, poly_prove_features.shape[0] - 1)
        print(f"Row: {row} / {poly_prove_features.shape[0]}")
        status, pred_label = sgd.inference(
            poly_prove_features.iloc[row:row + 1], prove_labels.iloc[row],
            mission=Missions.REG, error_thresholds=(curr_rmse, curr_rmse * 2),
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
        pprint(sgd.coefficient)
        print(f"Intercept: {sgd.intercept}")
        """
        ****************************************************************
        Linear Regression
        ----------------------------------------------------------------
        Coefficients:
        array([  1.29605856, -22.07030134,  39.21426325,  20.88183981,
               -34.69653007,  17.9623933 ,   6.6868702 ,  19.84714902,
                39.67536732,   5.45957327])
        Intercept: 152.31351288142994
        ****************************************************************
        ****************************************************************
        SGD Regression
        ----------------------------------------------------------------
        Coefficients:
        array([  1.5057529 , -20.64814107,  39.59393777,  20.22980277,
                -8.32047419,  -1.7355653 ,  -6.70807638,  13.37361374,
                30.44948492,   5.55308505])
        Intercept: 153.10458137874136
        ****************************************************************
        """

        sgd.save(model_name="diabetes_sgd_reg")


if __name__ == "__main__":
    main()
