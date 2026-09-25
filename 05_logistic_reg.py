#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 14:30
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   05_logistic_reg.py
# @Desc     :

from random import randint

from pandas import DataFrame, Series
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    FeaturesRobustScaler,
    LogisticSolvers,
    Missions,
    diagnose_cls_fit,
    expand_polynomial_features,
    get_cls_labels_distribution,
    split_data,
    tune_optimal_cls_degree,
)
from utils.ml.estimators import LogisticRegClassifier


def init_cancer() -> Bunch:
    """ Initialise the Breast Cancer Binary Classification dataset. """
    return load_breast_cancer()


def main() -> None:
    """ Main Function """
    cancer: Bunch = init_cancer()

    labels: Series = Series(cancer.target)
    get_cls_labels_distribution(labels, display=True)

    features: DataFrame = DataFrame(cancer.data, columns=cancer.feature_names)

    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        mission=Missions.CLS,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesRobustScaler(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)

        classifier = LogisticRegClassifier(
            penalty_strength=1.0,
            l1_ratio=0.5,
            solver=LogisticSolvers.SAGA,
            epochs=5_000,
        )

        degrees: list[int] = [1, 2, 3]
        best_degree, best_f1 = tune_optimal_cls_degree(
            classifier,
            train_features=train_features, train_labels=train_labels,
            valid_features=valid_features, valid_labels=valid_labels,
            degrees=degrees, display=True,
        )
        """
        ****************************************************************
        The function named 'tune_optimal_cls_degree' is starting:
        ----------------------------------------------------------------
        Polynomial Degree: 1, Weighted F1-Score: 0.9766
        Polynomial Degree: 2, Weighted F1-Score: 0.9765
        Polynomial Degree: 3, Weighted F1-Score: 0.9766
        ----------------------------------------------------------------
        Best Polynomial Degree: 1, Best F1-Score: 0.9766
        ----------------------------------------------------------------
        The function named 'tune_optimal_cls_degree' took 7.7492 seconds to complete.
        ****************************************************************
        """
        poly_train_features, poly_valid_features, poly_prove_features = expand_polynomial_features(
            best_degree,
            train_features=train_features,
            valid_features=valid_features,
            prove_features=prove_features
        )

        classifier.train(poly_train_features, train_labels)

        train_predictions = classifier.predict(poly_train_features)
        train_metrics = classifier.eval_cls(train_labels, train_predictions, display=False)
        train_f1: float = train_metrics["f1_score"]

        valid_predictions = classifier.predict(poly_valid_features)
        valid_metrics = classifier.eval_cls(valid_labels, valid_predictions, display=True)
        valid_f1: float = valid_metrics["f1_score"]
        """
        ****************************************************************
        Classification Evaluation Metrics
        ----------------------------------------------------------------
        Accuracy  : 0.9765
        Precision : 0.9779
        Recall    : 0.9765
        F1_score  : 0.9766
        ****************************************************************
        """

        diagnose_cls_fit(train_f1, valid_f1, display=True)
        """
        ****************************************************************
        The function named 'diagnose_cls_fit' is starting:
        ----------------------------------------------------------------
        Model Diagnosis: Well-Fitted! (Good generalization performance)
        ----------------------------------------------------------------
        The function named 'diagnose_cls_fit' took 0.0000 seconds to complete.
        ****************************************************************
        """

        row: int = randint(0, poly_prove_features.shape[0] - 1)
        print(f"Row: {row} / {poly_prove_features.shape[0]}")
        status, pred_label = classifier.inference(
            poly_prove_features.iloc[row:row + 1], prove_labels.iloc[row],
            mission=Missions.CLS,
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
