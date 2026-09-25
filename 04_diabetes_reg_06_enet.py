#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 13:56
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   04_diabetes_reg_06_enet.py
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
    diagnose_reg_fit,
    expand_polynomial_features,
    get_reg_labels_distribution,
    split_data,
    tune_optimal_reg_degree,
)
from utils.ml.estimators import ElasticNetReg


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

        enet = ElasticNetReg(l1_ratio=0.5)
        degrees: list[int] = [1, 2, 3]
        best_degree, best_rmse = tune_optimal_reg_degree(
            enet,
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

        enet.train(poly_train_features, train_labels)

        train_predictions = enet.predict(poly_train_features)
        train_metrics = enet.eval_reg(train_labels, train_predictions, display=False)
        train_rmse: float = train_metrics["rmse"]

        valid_predictions = enet.predict(poly_valid_features)
        valid_metrics = enet.eval_reg(valid_labels, valid_predictions, display=True)
        valid_rmse: float = valid_metrics["rmse"]
        valid_r2: float = valid_metrics["r2"]
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
        ****************************************************************
        Regression Evaluation Metrics - Lasso Regression
        ----------------------------------------------------------------
        R² Score  : 0.3593
        RMSE      : 51.2272
        MAE       : 42.5607
        MSE       : 2624.2272
        MAPE      : 35.5737%
        ****************************************************************
        ****************************************************************
        Regression Evaluation Metrics - Ridge Regression
        ----------------------------------------------------------------
        R² Score  : 0.3551
        RMSE      : 51.3955
        MAE       : 42.3887
        MSE       : 2641.4929
        MAPE      : 35.3699%
        ****************************************************************
        ****************************************************************
        Regression Evaluation Metrics - ElasticNet Regression
        ----------------------------------------------------------------
        R² Score  : 0.3508
        RMSE      : 51.5676
        MAE       : 42.9254
        MSE       : 2659.2194
        MAPE      : 37.2828%
        ****************************************************************
        """
        diagnose_reg_fit(train_rmse, valid_rmse, valid_r2, float(train_labels.std()), display=True)
        """
        ****************************************************************
        The function named 'diagnose_fit' is starting:
        ----------------------------------------------------------------
        Model Diagnosis: Well-Fitted! (Good generalization performance)
        ----------------------------------------------------------------
        The function named 'diagnose_fit' took 0.0000 seconds to complete.
        ****************************************************************
        """

        curr_rmse: float = valid_metrics.get("rmse", best_rmse)

        row: int = randint(0, poly_prove_features.shape[0] - 1)
        print(f"Row: {row} / {poly_prove_features.shape[0]}")
        status, pred_label = enet.inference(
            poly_prove_features.iloc[row:row + 1], prove_labels.iloc[row],
            mission=Missions.REG, reg_error_thresholds=(curr_rmse, curr_rmse * 2),
            display=False
        )
        if status:
            print(
                f"{green('Acceptable')}! "
                f"Prediction: {pred_label}, "
                f"Label: {prove_labels.iloc[row]}."
            )
            print()
        else:
            print(
                f"{red('Unacceptable')}! "
                f"Prediction: {pred_label}, "
                f"Label: {prove_labels.iloc[row]}."
            )
            print()

        print("Coefficients:")
        pprint(enet.coefficient)
        print(f"Intercept: {enet.intercept}")
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
        ****************************************************************
        Lasso Regression
        ----------------------------------------------------------------
        Coefficients:
        array([  0.        , -14.28556669,  38.66637635,  17.77635466,
                -1.22420314,  -0.        , -12.84309545,   0.        ,
                30.24573908,   4.77711586])
        Intercept: 150.9204280878722
        ****************************************************************
        ****************************************************************
        Ridge Regression
        ----------------------------------------------------------------
        Coefficients:
        array([  1.42308976, -21.57443509,  39.01082693,  20.66903534,
               -21.24926135,   7.96380094,  -0.24013886,  16.6002452 ,
                34.71886888,   5.61362946])
        Intercept: 152.94340110917997
        ****************************************************************
        ****************************************************************
        ElasticNet Regression
        ----------------------------------------------------------------
        Coefficients:
        array([ 2.37994883, -3.99016853, 21.04591863, 12.49203521,  1.81836486,
                0.        , -9.68777858,  7.65477409, 18.1950087 ,  8.04442478])
        Intercept: 148.74706002482063
        ****************************************************************
        """


if __name__ == "__main__":
    main()
