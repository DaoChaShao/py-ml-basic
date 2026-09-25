#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/25 18:53
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_tree_03_gbdt_01_cls.py
# @Desc     :   

from random import randint

from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from utils import red, green
from utils.ml import (
    diagnose_cls_fit,
    expand_polynomial_features,
    FeaturesTransformer,
    get_cls_labels_distribution,
    Missions,
    split_data,
    summary_dataframe,
    TitanicFeatures,
    TreeClsLoss,
    tune_optimal_cls_degree,
)
from utils.ml.estimators import GBDTree


def init_titanic() -> Bunch:
    """
    Initialise the Titanic dataset.

    :return: The Titanic dataset.
    """
    return fetch_openml("titanic", version=1, as_frame=True)


def main() -> None:
    """ Main Function """
    titanic: Bunch = init_titanic()
    features = titanic.data
    labels = titanic.target
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

        # tree = GBDTree(Missions.CLS, TreeClsLoss.LOG_LOSS, max_depth=4, min_samples_leaf=5)
        tree = GBDTree(
            Missions.CLS,
            TreeClsLoss.LOG_LOSS,
            learning_rate=0.01, n_estimators=500,
            max_depth=4, min_samples_leaf=5
        )

        degrees: list[int] = [1, 2, 3]
        best_degree, best_f1 = tune_optimal_cls_degree(
            tree,
            train_features=transformed_train_features, train_labels=train_labels,
            valid_features=transformed_valid_features, valid_labels=valid_labels,
            degrees=degrees, display=True,
        )
        """
        ****************************************************************
        The function named 'tune_optimal_cls_degree' is starting (Decision Tree):
        ----------------------------------------------------------------
        Polynomial Degree: 1, Weighted F1-Score: 0.7900
        Polynomial Degree: 2, Weighted F1-Score: 0.7625
        Polynomial Degree: 3, Weighted F1-Score: 0.7934
        ----------------------------------------------------------------
        Best Polynomial Degree: 3, Best F1-Score: 0.7934
        ----------------------------------------------------------------
        The function named 'tune_optimal_cls_degree' took 0.0250 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'tune_optimal_cls_degree' is starting (Random Forest):
        ----------------------------------------------------------------
        Polynomial Degree: 1, Weighted F1-Score: 0.7735
        Polynomial Degree: 2, Weighted F1-Score: 0.7886
        Polynomial Degree: 3, Weighted F1-Score: 0.8071
        ----------------------------------------------------------------
        Best Polynomial Degree: 3, Best F1-Score: 0.8071
        ----------------------------------------------------------------
        The function named 'tune_optimal_cls_degree' took 0.3105 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'tune_optimal_cls_degree' is starting (GBDT(lr = 0.1)):
        ----------------------------------------------------------------
        Polynomial Degree: 1, Weighted F1-Score: 0.8044
        Polynomial Degree: 2, Weighted F1-Score: 0.7728
        Polynomial Degree: 3, Weighted F1-Score: 0.7927
        ----------------------------------------------------------------
        Best Polynomial Degree: 1, Best F1-Score: 0.8044
        ----------------------------------------------------------------
        The function named 'tune_optimal_cls_degree' took 1.0721 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'tune_optimal_cls_degree' is starting (GBDT(lr = 0.01)):
        ----------------------------------------------------------------
        Polynomial Degree: 1, Weighted F1-Score: 0.8244
        Polynomial Degree: 2, Weighted F1-Score: 0.7815
        Polynomial Degree: 3, Weighted F1-Score: 0.7975
        ----------------------------------------------------------------
        Best Polynomial Degree: 1, Best F1-Score: 0.8244
        ----------------------------------------------------------------
        The function named 'tune_optimal_cls_degree' took 1.0611 seconds to complete.
        ****************************************************************
        """

        poly_train_features, poly_valid_features, poly_prove_features = expand_polynomial_features(
            best_degree,
            train_features=transformed_train_features,
            valid_features=transformed_valid_features,
            prove_features=transformed_prove_features
        )

        tree.train(poly_train_features, train_labels)

        train_predictions = tree.predict(poly_train_features)
        train_metrics = tree.eval_cls(train_labels, train_predictions, display=False)
        train_f1: float = train_metrics["f1_score"]

        valid_predictions = tree.predict(poly_valid_features)
        valid_metrics = tree.eval_cls(valid_labels, valid_predictions, display=True)
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
        """

        diagnose_cls_fit(train_f1, valid_f1, display=True)

        row: int = randint(0, poly_prove_features.shape[0] - 1)
        print(f"Row: {row} / {poly_prove_features.shape[0]}")
        status, pred_label = tree.inference(
            poly_prove_features.iloc[row:row + 1], prove_labels.iloc[row],
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
