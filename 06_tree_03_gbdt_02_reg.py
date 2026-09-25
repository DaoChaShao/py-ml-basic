#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/25 20:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_tree_03_gbdt_02_reg.py
# @Desc     :

from pprint import pprint
from random import randint

from pandas import DataFrame, Series
from sklearn.datasets import load_diabetes
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    Missions,
    TreeRegCriteria,
    diagnose_reg_fit,
    get_reg_labels_distribution,
    split_data,
    tune_optimal_gbd_tree,
)
from utils.ml.estimators import GBDTree


def init_diabetes() -> Bunch:
    """
    Initialise the diabetes dataset.

    :return: The diabetes dataset.
    """
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

    learning_rates = [0.01, 0.03, 0.05, 0.1]
    estimators = [100, 200, 300]
    deeps = [2, 3, 4, 5]
    leafs = [1, 2, 5, 10]
    splits = [2, 5, 10, 20]
    best_learning_rate, best_n_estimators, best_depth, best_split, best_leaf, best_rmse = tune_optimal_gbd_tree(
        GBDTree,
        train_features=train_features,
        train_labels=train_labels,
        valid_features=valid_features,
        valid_labels=valid_labels,
        criterion=TreeRegCriteria.SQUARED_ERROR,
        learning_rates=learning_rates,
        n_estimators_list=estimators,
        max_depths=deeps,
        min_samples_splits=splits,
        min_samples_leafs=leafs,
        display=True
    )
    """
    ****************************************************************
    The function named 'tune_optimal_reg_tree' is starting (Decision Tree):
    ----------------------------------------------------------------
    Best max_depth=05, Best min_samples_leaf=10, Best min_samples_split=02, Best RMSE=65.3399
    ----------------------------------------------------------------
    The function named 'tune_optimal_reg_tree' took 0.5869 seconds to complete.
    ****************************************************************
    ****************************************************************
    The function named 'tune_optimal_reg_tree' is starting (Random Forest):
    ----------------------------------------------------------------
    Best max_depth=04, Best min_samples_leaf=15, Best min_samples_split=02, Best RMSE=54.0371
    ----------------------------------------------------------------
    The function named 'tune_optimal_reg_tree' took 18.6653 seconds to complete.
    ****************************************************************
     ****************************************************************
    The function named 'tune_optimal_reg_tree' is starting (GBDT):
    ----------------------------------------------------------------
    Best learning_rate=0.01, Best n_estimators=100, Best max_depth=02, Best min_samples_leaf=01, Best min_samples_split=02, Best RMSE=56.3769
    ----------------------------------------------------------------
    The function named 'tune_optimal_gbd_tree' took 74.2828 seconds to complete.
    ****************************************************************
    """

    tree = GBDTree(
        Missions.REG,
        TreeRegCriteria.SQUARED_ERROR,
        learning_rate=best_learning_rate,
        n_estimators=best_n_estimators,
        max_depth=best_depth,
        min_samples_split=best_split,
        min_samples_leaf=best_leaf
    )
    tree.train(train_features, train_labels)

    train_predictions = tree.predict(train_features)
    train_metrics = tree.eval_reg(train_labels, train_predictions, display=False)
    train_rmse: float = train_metrics["rmse"]

    valid_predictions = tree.predict(valid_features)
    valid_metrics = tree.eval_reg(valid_labels, valid_predictions, display=True)
    valid_rmse: float = valid_metrics["rmse"]
    valid_r2: float = valid_metrics["r2"]
    """
    ****************************************************************
    Regression Evaluation Metrics
    ----------------------------------------------------------------
    R² Score  : -0.0423
    RMSE      : 65.3399
    MAE       : 52.3079
    MSE       : 4269.3056
    MAPE      : 39.7905%
    ****************************************************************
    ****************************************************************
    Regression Evaluation Metrics
    ----------------------------------------------------------------
    R² Score  : 0.2871
    RMSE      : 54.0371
    MAE       : 44.3425
    MSE       : 2920.0120
    MAPE      : 37.9474%
    ****************************************************************
    ****************************************************************
    Regression Evaluation Metrics
    ----------------------------------------------------------------
    R² Score  : 0.2241
    RMSE      : 56.3769
    MAE       : 48.3764
    MSE       : 3178.3498
    MAPE      : 41.9400%
    ****************************************************************
    """

    diagnose_reg_fit(train_rmse, valid_rmse, valid_r2, float(train_labels.std()), display=True)

    curr_rmse: float = valid_metrics.get("rmse", best_rmse)

    row: int = randint(0, prove_features.shape[0] - 1)
    print(f"Row: {row} / {prove_features.shape[0]}")
    status, pred_label = tree.inference(
        prove_features.iloc[row:row + 1], prove_labels.iloc[row],
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

    print("Feature Importances:")
    pprint(tree.feature_importances)


if __name__ == "__main__":
    main()
