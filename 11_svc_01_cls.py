#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/28 12:43
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   11_svc_01_cls.py
# @Desc     :   

from pprint import pprint
from random import randint

from pandas import DataFrame, Series
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import Bunch

from utils import red, green
from utils.ml import (
    FeaturesCategories,
    FeaturesRobustScaler,
    GMMCovarianceCategories,
    GMMInitParamsCategories,
    Missions,
    SVMKernelCategories,
    diagnose_cls_fit,
    eval_clustering_classification,
    eval_clustering_with_ch,
    get_cls_labels_distribution,
    split_data,
    summary_dataframe,
)
from utils.ml.estimators import HyperSVClassifier


def init_wine() -> Bunch:
    """ Initialize Wine Dataset """
    return load_wine()


def main() -> None:
    """ Main Function """
    wine: Bunch = init_wine()
    features = wine.data
    labels = wine.target
    # print(type(features), type(labels))
    features: DataFrame = DataFrame(features, columns=wine.feature_names)
    labels: Series = Series(labels)
    # print(type(features), type(labels))
    # print(features.head())
    # print(labels.head())

    summary_dataframe(features, display=False)

    get_cls_labels_distribution(labels, display=True)

    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        mission=Missions.CLS,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesRobustScaler(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)

        svc = HyperSVClassifier(0.1)

        # GridSearchCV
        cross_validation: KFold = KFold(n_splits=5, shuffle=True, random_state=27)
        search = GridSearchCV(
            svc,
            {
                "regularization_param": [1.0, 0.1, 0.01],
                "kernel": [SVMKernelCategories.RBF, SVMKernelCategories.POLY, SVMKernelCategories.LINEAR],
                "degree": [2, 3, 4],
            },
            scoring="accuracy",
            cv=cross_validation,
            n_jobs=-1,  # -1: Use all available CPU cores | 1: Use 1 CPU core
            verbose=1,  # Set to 1 to print whole progress
        )

        search.fit(train_features, train_labels)
        print("Best Parameters:")
        pprint(search.best_params_)
        print("Best CV Score:", search.best_score_)
        """
        ****************************************************************
        Best Parameters:
        {'degree': 2,
         'kernel': <SVMKernelCategories.LINEAR: 'linear'>,
         'regularization_param': 0.1}
         ----------------------------------------------------------------
        Best CV Score: 0.976
        ****************************************************************
        """

        svc = search.best_estimator_
        train_predictions = search.predict(train_features)
        train_metrics = svc.eval_cls(train_labels, train_predictions, display=False)
        train_f1: float = train_metrics["f1_score"]

        valid_predictions = search.predict(valid_features)
        valid_metrics = svc.eval_cls(valid_labels, valid_predictions, display=True)
        valid_f1: float = valid_metrics["f1_score"]
        """
        ****************************************************************
        Classification Evaluation Metrics
        ----------------------------------------------------------------
        Accuracy  : 1.0000
        Precision : 1.0000
        Recall    : 1.0000
        F1_score  : 1.0000
        ****************************************************************
        """

        diagnose_cls_fit(train_f1, valid_f1, display=True)

        row: int = randint(0, prove_features.shape[0] - 1)
        print(f"Row: {row} / {prove_features.shape[0]}")
        status, pred_label = search.best_estimator_.inference(
            prove_features.iloc[row:row + 1], prove_labels.iloc[row],
            mission=Missions.CLS,
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


if __name__ == "__main__":
    main()
