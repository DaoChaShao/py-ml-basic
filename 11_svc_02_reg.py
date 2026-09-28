#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/28 13:13
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   11_svc_02_reg.py
# @Desc     :

from pprint import pprint
from random import randint

from pandas import DataFrame, Series
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    FeaturesRobustScaler,
    Missions,
    SVMKernelCategories,
    diagnose_reg_fit,
    get_reg_labels_distribution,
    split_data,
)
from utils.ml.estimators import HyperSVRegressor


def init_diabetes() -> Bunch:
    """ Initialise the diabetes dataset. """
    return load_diabetes()


def main() -> None:
    """ Main Function """
    diabetes = init_diabetes()
    features: DataFrame = DataFrame(diabetes.data, columns=diabetes.feature_names)
    labels: Series = Series(diabetes.target)

    get_reg_labels_distribution(labels, display=True)

    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        mission=Missions.REG,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesRobustScaler(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)

        svr = HyperSVRegressor()

        # GridSearchCV
        cross_validation: KFold = KFold(n_splits=5, shuffle=True, random_state=27)
        search = GridSearchCV(
            svr,
            {

                "kernel": [SVMKernelCategories.RBF, SVMKernelCategories.POLY, SVMKernelCategories.LINEAR],
                "degree": [2, 3, 4],
                "regularization_param": [1.0, 0.1, 0.01],
                "epsilon": [0.1, 0.01, 0.001],
            },
            scoring="r2",
            cv=cross_validation,
            n_jobs=-1,  # -1: Use all available CPU cores | 1: Use 1 CPU core
            verbose=1,  # Set to 1 to print whole progress
        )
        """
        ****************************************************************
        Best Parameters:
        {'degree': 2,
         'epsilon': 0.001,
         'kernel': <SVMKernelCategories.LINEAR: 'linear'>,
         'regularization_param': 1.0}
         ----------------------------------------------------------------
        Best CV Score: 0.4656032339871965
        ****************************************************************
        """

        search.fit(train_features, train_labels)
        print("Best Parameters:")
        pprint(search.best_params_)
        print("Best CV Score:", search.best_score_)

        svr = search.best_estimator_
        train_predictions = svr.predict(train_features)
        train_metrics = svr.eval_reg(train_labels, train_predictions, display=False)
        train_rmse: float = train_metrics["rmse"]

        valid_predictions = svr.predict(valid_features)
        valid_metrics = svr.eval_reg(valid_labels, valid_predictions, display=True)
        valid_rmse: float = valid_metrics["rmse"]
        valid_r2: float = valid_metrics["r2"]
        """
        ****************************************************************
        Regression Evaluation Metrics (Decision Tree)
        ----------------------------------------------------------------
        R² Score  : -0.0423
        RMSE      : 65.3399
        MAE       : 52.3079
        MSE       : 4269.3056
        MAPE      : 39.7905%
        ****************************************************************
        ****************************************************************
        Regression Evaluation Metrics (Random Forest)
        ----------------------------------------------------------------
        R² Score  : 0.2871
        RMSE      : 54.0371
        MAE       : 44.3425
        MSE       : 2920.0120
        MAPE      : 37.9474%
        ****************************************************************
        ****************************************************************
        Regression Evaluation Metrics (Support Vector Machine Regressor)
        ----------------------------------------------------------------
        R² Score  : 0.4064
        RMSE      : 49.3101
        MAE       : 40.4791
        MSE       : 2431.4907
        MAPE      : 33.1308%
        ****************************************************************
        """

        diagnose_reg_fit(train_rmse, valid_rmse, valid_r2, float(train_labels.std()), display=True)

        row: int = randint(0, prove_features.shape[0] - 1)
        print(f"Row: {row} / {prove_features.shape[0]}")
        status, pred_label = svr.inference(
            prove_features.iloc[row:row + 1], prove_labels.iloc[row],
            mission=Missions.REG, reg_error_thresholds=(valid_rmse, valid_rmse * 2),
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


if __name__ == "__main__":
    main()
