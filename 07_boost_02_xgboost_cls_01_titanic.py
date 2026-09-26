#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/26 17:58
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   07_boost_02_xgboost_cls_01_titanic.py
# @Desc     :   

from pandas import DataFrame, Series
from random import randint

from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    FeaturesTransformer,
    Missions,
    TitanicFeatures,
    diagnose_cls_fit,
    get_cls_labels_distribution,
    split_data,
    summary_dataframe,
    XGBClsObjectives,
)
from utils.ml.estimators import HyperXGBooster


def init_titanic() -> Bunch:
    """
    Initialise the Titanic dataset.

    :return: The Titanic dataset.
    """
    return fetch_openml("titanic", version=1, as_frame=True)


def main() -> None:
    """ Main Function """
    titanic: Bunch = init_titanic()
    features: DataFrame = DataFrame(titanic.data, columns=titanic.feature_names)
    labels: Series = Series(titanic.target.astype(int))
    print(type(features), type(labels))

    get_cls_labels_distribution(labels, display=True)

    summary_dataframe(features)
    drop_cols: list[TitanicFeatures] = [
        TitanicFeatures.BOAT.EN,
        TitanicFeatures.BODY.EN,
        TitanicFeatures.CABIN.EN,
        TitanicFeatures.HOME_DEST.EN,
        TitanicFeatures.NAME.EN,
        TitanicFeatures.TICKET.EN,
    ]
    features = features.drop(columns=drop_cols)
    print(features.columns.to_list())

    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        mission=Missions.CLS,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesTransformer(train_features) as transformer:
        transformed_train_features = transformer.transform(train_features)
        transformed_valid_features = transformer.transform(valid_features)
        transformed_prove_features = transformer.transform(prove_features)

        adaboost = HyperXGBooster(Missions.CLS, XGBClsObjectives.BINARY_LOGISTIC)
        cross_validation: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
        search = GridSearchCV(
            adaboost,
            {
                "learning_rate": [0.03, 0.1],
                "n_estimators": [100, 200],
                "max_depth": [2, 3, 4],
            },
            scoring="accuracy",
            cv=cross_validation,
            n_jobs=1,  # -1: Use all available CPU cores | 1: Use 1 CPU core
            verbose=1,  # Set to 1 to print whole progress
        )

        search.fit(transformed_train_features, train_labels)
        print("Best Parameters:", search.best_params_)
        print("Best CV Score:", search.best_score_)

        train_predictions = search.predict(transformed_train_features)
        train_metrics = search.best_estimator_.eval_cls(train_labels, train_predictions, display=False)
        train_f1: float = train_metrics["f1_score"]

        valid_predictions = search.predict(transformed_valid_features)
        valid_metrics = search.best_estimator_.eval_cls(valid_labels, valid_predictions, display=True)
        valid_f1: float = valid_metrics["f1_score"]
        """
        ****************************************************************
        Classification Evaluation Metrics - Decision Tree
        ----------------------------------------------------------------
        Accuracy  : 0.7959
        Precision : 0.7938
        Recall    : 0.7959
        F1_score  : 0.7934
        ****************************************************************
        ****************************************************************
        Classification Evaluation Metrics - Random Forest
        ----------------------------------------------------------------
        Accuracy  : 0.8112
        Precision : 0.8108
        Recall    : 0.8112
        F1_score  : 0.8071
        ****************************************************************
        ****************************************************************
        Classification Evaluation Metrics - GBDT(lr = 0.1)
        ----------------------------------------------------------------
        Accuracy  : 0.8061
        Precision : 0.8043
        Recall    : 0.8061
        F1_score  : 0.8044
        ****************************************************************
        ****************************************************************
        Classification Evaluation Metrics - GBDT(lr = 0.01)
        ----------------------------------------------------------------
        Accuracy  : 0.8265
        Precision : 0.8253
        Recall    : 0.8265
        F1_score  : 0.8244
        ****************************************************************
        ****************************************************************
        Classification Evaluation Metrics - HyperGBDT
        ----------------------------------------------------------------
        Accuracy  : 0.8214
        Precision : 0.8199
        Recall    : 0.8214
        F1_score  : 0.8196
        ****************************************************************
        ****************************************************************
        Classification Evaluation Metrics - XGBoost
        ----------------------------------------------------------------
        Accuracy  : 0.8214
        Precision : 0.8202
        Recall    : 0.8214
        F1_score  : 0.8189
        ****************************************************************
        """

        diagnose_cls_fit(train_f1, valid_f1, display=True)

        row: int = randint(0, transformed_prove_features.shape[0] - 1)
        print(f"Row: {row} / {transformed_prove_features.shape[0]}")
        status, pred_label = search.best_estimator_.inference(
            transformed_prove_features.iloc[row:row + 1], prove_labels.iloc[row],
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
