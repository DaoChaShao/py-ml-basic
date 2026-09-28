#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/28 23:28
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   12_anomaly_02_lof.py
# @Desc     :

from pprint import pprint
from random import randint
from typing import Any

from numpy import concatenate
from pandas import DataFrame, Series
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from utils import green, lines, red, timer
from utils.ml import (
    AveStrategies,
    DistanceMetrics,
    FeaturesRobustScaler,
    LOFAlgorithms,
    Missions,
    diagnose_cls_fit,
    get_cls_labels_distribution,
    split_data,
    summary_dataframe,
)
from utils.ml.estimators import HyperLOF


@timer
def generate_normals(samples: int = 500, *, randomness: int = 27, display: bool = True) -> DataFrame:
    """
    Generate normal features.

    :param samples: Number of samples to generate.
    :param randomness: Random state for reproducibility.
    :param display: Whether to display the generated features.
    :return: Generated features.
    """
    _features, _ = make_blobs(
        n_samples=samples,
        centers=1,
        cluster_std=1.0,
        random_state=randomness,
    )
    _features: DataFrame = DataFrame(_features, columns=["x", "y"])

    if display:
        print(f"Generated {samples} normal features with randomness {randomness}")
        lines()
        print(_features.head())
    return _features


@timer
def generate_anomalies(*, samples: int = 500, randomness: int = 27, display: bool = True) -> DataFrame:
    """
    Generate anomaly features.

    :param samples: Number of samples to generate.
    :param randomness: Random state for reproducibility.
    :param display: Whether to display the generated features.
    :return: Generated features.
    """
    _features, _ = make_blobs(
        n_samples=samples,
        centers=[[10, 10]],
        cluster_std=0.5,
        random_state=randomness,
    )
    _features: DataFrame = DataFrame(_features, columns=["x", "y"])

    if display:
        print(f"Generated {samples} anomaly features with randomness {randomness}")
        lines()
        print(_features.head())
    return _features


# @timer
# def generate_normals(*, samples: int = 500, randomness: int = 27, display: bool = True) -> DataFrame:
#     """
#     Generate LOF normal features.
#
#     :param samples: Number of samples to generate.
#     :param randomness: Random state for reproducibility.
#     :param display: Whether to display the generated features.
#     :return: Generated features.
#     """
#     _features, _ = make_blobs(
#         n_samples=samples,
#         centers=[[10, 10]],
#         cluster_std=1.0,
#         random_state=randomness,
#     )
#     _features: DataFrame = DataFrame(_features, columns=["x", "y"])
#
#     if display:
#         print(f"Generated {samples} LOF normal features with randomness {randomness}")
#         lines()
#         print(_features.head())
#     return _features


# @timer
# def generate_lof_anomalies(*, samples: int = 500, randomness: int = 27, display: bool = True) -> DataFrame:
#     """
#     Generate LOF anomaly features.
#
#     :param samples: Number of samples to generate.
#     :param randomness: Random state for reproducibility.
#     :param display: Whether to display the generated features.
#     :return: Generated features.
#     """
#     _features, _ = make_blobs(
#         n_samples=samples,
#         centers=[[10, 10]],
#         cluster_std=2.5,
#         random_state=randomness,
#     )
#     _features: DataFrame = DataFrame(_features, columns=["x", "y"])
#
#     if display:
#         print(f"Generated {samples} anomaly features with randomness {randomness}")
#         lines()
#         print(_features.head())
#     return _features

# @timer
# def generate_anomalies(*, samples: int = 50, randomness: int = 27, display: bool = True) -> DataFrame:
#     _rng = default_rng(randomness)
#
#     _angles = _rng.uniform(low=0.0, high=2.0 * pi, size=samples)
#     _radius: float = 3.5
#
#     _x = 10 + _radius * cos(_angles)
#     _y = 10 + _radius * sin(_angles)
#     _features = DataFrame({"x": _x, "y": _y})
#
#     if display:
#         print(f"Generated {samples} LOF anomaly features with randomness {randomness}")
#         lines()
#         print(_features.head())
#     return _features


@timer
def count_labels_distribution(labels: Series, *, display: bool = True) -> dict[Any, dict[str, Any]]:
    """
    Count and display the distribution of labels.

    :param labels: The labels to count and display.
    :param display: Whether to display the distribution.
    :return: A dictionary containing the counts and ratios of each label.
    """
    _values = labels.value_counts()

    _counts: dict[Any, int] = {}
    for label, amount in _values.items():
        _counts[label] = amount

    _ratios: dict[Any, dict[str, Any]] = {}
    _total = sum(_counts.values())
    for label, amount in _counts.items():
        ratio = round(amount / _total, 4)
        _ratios[label] = {"amount": amount, "ratio": ratio}

        if display:
            print(f"Label: {label}, Amount: {amount}, Ratio: {ratio:.2%}")
    return _ratios


def main() -> None:
    """ Main Function """
    nor_features: DataFrame = generate_normals(samples=950)
    anor_features: DataFrame = generate_anomalies(samples=50)

    # Create and concatenate features within normal and anomaly dataframes
    features: DataFrame = DataFrame(concatenate([nor_features, anor_features], axis=0), columns=["x", "y"])
    # Create labels
    labels: Series = Series(concatenate([[1] * len(nor_features), [-1] * len(anor_features)]), name="label")
    print(type(features), type(labels), end="\n\n")

    get_cls_labels_distribution(labels, display=True)

    summary_dataframe(features, display=True)

    count_labels_distribution(labels, display=True)

    anor_amount: int = labels.value_counts().get(-1, 0)
    ratio: float = anor_amount / len(labels)

    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        mission=Missions.CLS,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesRobustScaler(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)
        print(type(train_features), type(valid_features), type(prove_features), end="\n\n")

        lof = HyperLOF(20)

        # GridSearchCV
        scoring = {
            "precision": make_scorer(precision_score, pos_label=-1, zero_division=0, ),
            "recall": make_scorer(recall_score, pos_label=-1, zero_division=0, ),
            "f1": make_scorer(f1_score, pos_label=-1, zero_division=0, ),
        }
        cross_validation: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
        search = GridSearchCV(
            lof,
            {
                "n_neighbors": [10, 20, 35, 50],
                "algorithm": [LOFAlgorithms.BALL_TREE.value, LOFAlgorithms.BRUTE.value, LOFAlgorithms.KD_TREE.value],
                "leaf_size": [30, 50],
                "p": [2],
                "metric": [
                    DistanceMetrics.CHEBYSHEV.value,
                    DistanceMetrics.EUCLIDEAN.value,
                    DistanceMetrics.MANHATTAN.value,
                    DistanceMetrics.MINKOWSKI.value,
                ],
                "contamination": [ratio],
            },
            scoring=scoring,
            refit="f1",
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
        {'algorithm': 'ball_tree',
         'contamination': np.float64(0.05),
         'leaf_size': 30,
         'metric': 'manhattan',
         'n_neighbors': 50,
         'p': 2}
         ----------------------------------------------------------------
        Best CV Score: 0.9034032634032634
        ****************************************************************
        """

        lof = search.best_estimator_
        train_predictions = search.predict(train_features)
        train_metrics = lof.eval_cls(
            train_labels, train_predictions,
            ave_strategy=AveStrategies.BINARY, pos_label=-1, display=False
        )
        train_f1: float = train_metrics["f1_score"]

        valid_predictions = search.predict(valid_features)
        valid_metrics = lof.eval_cls(
            valid_labels, valid_predictions,
            ave_strategy=AveStrategies.BINARY, pos_label=-1, display=True
        )
        valid_f1: float = valid_metrics["f1_score"]
        """
        ****************************************************************
        Classification Evaluation Metrics - Distance
        ----------------------------------------------------------------
        Accuracy  : 0.9867
        Precision : 0.8750
        Recall    : 0.8750
        F1_score  : 0.8750
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
            print(f"{green('Correct')}! Prediction: {pred_label}, Label: {prove_labels.iloc[row]}.", end="\n\n")
        else:
            print(f"{red('Incorrect')}! Prediction: {pred_label}, Label: {prove_labels.iloc[row]}.", end="\n\n")


if __name__ == "__main__":
    main()
