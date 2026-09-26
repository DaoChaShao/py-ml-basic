#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/25 22:29
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   07_boost_01_adaboost_cls_02_iris.py
# @Desc     :

from pprint import pprint
from random import randint, uniform

from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

from utils import green, red
from utils.ml import (
    FeaturesStandardiser,
    IrisFeatures,
    IrisLabels,
    Missions,
    diagnose_cls_fit,
    expand_polynomial_features,
    split_data,
    summary_dataframe,
    tune_optimal_cls_degree,
)
from utils.ml.estimators import AdaBoost


@validate_call
def init_iris(top_n: int = Field(5, gt=0, le=150), *, display: bool = False, describe: bool = False) -> Bunch:
    """
    Initialise Iris Dataset

    :param top_n: Number of samples to display
    :param describe: Whether to describe the data and metadata
    :param display: Whether to display the data and metadata
    :return: Iris Dataset
    """
    iris = load_iris()
    if display:
        pprint(iris.data[:top_n])
        pprint(iris.target[:top_n])
        print()
        print(f"Iris Dataset Features Name: {iris.feature_names}")
        print(f"Iris Dataset Labels Name: {iris.target_names}")
        print(f"Iris Dataset Features Shape: {iris.data.shape}")
        print(f"Iris Dataset Labels Shape: {iris.target.shape}")
    if describe:
        print(f"Iris Dataset Description: {iris.DESCR}")
    return iris


def rand_a_sample(amount: int = 1, *, display: bool = False) -> DataFrame:
    """ Random a sample """
    _statistics: dict[IrisFeatures, tuple[float, float]] = {
        IrisFeatures.SEPAL_LENGTH: (4.3, 7.9),
        IrisFeatures.SEPAL_WIDTH: (2.0, 4.4),
        IrisFeatures.PETAL_LENGTH: (1.0, 6.9),
        IrisFeatures.PETAL_WIDTH: (0.1, 2.5),
    }

    _samples = []
    for _ in range(amount):
        _sample = [round(uniform(bounds[0], bounds[1]), 2) for bounds in _statistics.values()]
        _samples.append(_sample)

    _names = [feature.EN for feature in _statistics]
    dataframe: DataFrame = DataFrame(_samples, columns=_names)

    if display:
        print(dataframe)
    return dataframe


def main() -> None:
    """ Main Function """
    iris: Bunch = init_iris()
    features: DataFrame = DataFrame(iris.data, columns=iris.feature_names)
    labels: Series = Series(iris.target)

    summary_dataframe(features)

    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        mission=Missions.CLS,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesStandardiser(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)

        adaboost = AdaBoost(Missions.CLS)

        degrees: list[int] = [1, 2, 3]
        best_degree, best_f1 = tune_optimal_cls_degree(
            adaboost,
            train_features=train_features, train_labels=train_labels,
            valid_features=valid_features, valid_labels=valid_labels,
            degrees=degrees, display=True,
        )

        poly_train_features, poly_valid_features, poly_prove_features = expand_polynomial_features(
            best_degree,
            train_features=train_features,
            valid_features=valid_features,
            prove_features=prove_features
        )

        adaboost.fit(poly_train_features, train_labels)

        train_predictions = adaboost.predict(poly_train_features)
        train_metrics = adaboost.eval_cls(train_labels, train_predictions, display=False)
        train_f1: float = train_metrics["f1_score"]

        valid_predictions = adaboost.predict(poly_valid_features)
        valid_metrics = adaboost.eval_cls(valid_labels, valid_predictions, display=True)
        valid_f1: float = valid_metrics["f1_score"]
        """
        ****************************************************************
        Classification Evaluation Metrics - KNN
        ----------------------------------------------------------------
        Accuracy  : 0.9545
        Precision : 0.9602
        Recall    : 0.9545
        F1_score  : 0.9545
        ****************************************************************
        ****************************************************************
        Classification Evaluation Metrics - Adaboost
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
        status, pred_label = adaboost.inference(prove_features.iloc[[row]], prove_labels.iloc[row], display=False)
        if status:
            print(
                f"{green('Correct')}! "
                f"Prediction: {list(IrisLabels)[pred_label].CN}, "
                f"Label: {list(IrisLabels)[prove_labels.iloc[row]].CN}."
            )
        else:
            print(
                f"{red('Incorrect')}! "
                f"Prediction: {list(IrisLabels)[pred_label].CN}, "
                f"Label: {list(IrisLabels)[prove_labels.iloc[row]].CN}."
            )

        sample = rand_a_sample(amount=1, display=True)
        sample_features = standardiser.transform(sample)
        pred = adaboost.predict(sample_features)[0]
        print(f"Prediction: {list(IrisLabels)[pred].CN}")
        probs = adaboost.confidence(sample_features)[0]
        for label, prob in zip(IrisLabels, probs, strict=True):
            print(f"{label.CN:<6}: {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
