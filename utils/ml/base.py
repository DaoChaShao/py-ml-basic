#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/20 22:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base.py
# @Desc     :

from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, Literal

from pandas import DataFrame, Series
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..helper import Access
from ..highlighter import lines, stars
from .types import AveStrategies, GridSearchTonesResponse, KNNMissions, ScoreStrategies


class Base(ABC, Access):
    """ Base class for all machine learning models. """

    def __init__(self) -> None:
        """
        Initialize the base class.

        :return: None
        """
        super().__init__()

    @abstractmethod
    def train(self, features: DataFrame, labels: Series) -> None:
        """
        Train the model.

        :param features: The features to train on.
        :param labels: The labels to train on.
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, features: DataFrame) -> Any:
        """
        Predict the labels for the given features.

        :param features: The features to predict on.
        :return: The predicted labels.
        """
        pass

    @staticmethod
    def eval_cls(
            valid_labels: Series, predictions: Series,
            *,
            ave_strategy: str | AveStrategies | Literal[
                "weighted", "macro", "micro", "samples", "binary"
            ] = AveStrategies.WEIGHTED,
            display: bool = False
    ) -> dict[str, Any]:
        """
        Evaluate the _acc of the model.

        :param valid_labels: The true labels.
        :param predictions: The predicted labels.
        :param ave_strategy: The average method for multi-class metrics ('weighted', 'macro', etc.).
        :param display: Whether to print the formatted evaluation result.
        :return: Dictionary containing calculated evaluation metrics and matrices.
        """
        _acc: float = accuracy_score(valid_labels, predictions)
        _pre: float = precision_score(valid_labels, predictions, average=AveStrategies(ave_strategy), zero_division=0)
        _rec: float = recall_score(valid_labels, predictions, average=AveStrategies(ave_strategy), zero_division=0)
        _f1: float = f1_score(valid_labels, predictions, average=AveStrategies(ave_strategy), zero_division=0)
        _metrics = {
            "accuracy": _acc,
            "precision": _pre,
            "recall": _rec,
            "f1_score": _f1,
        }

        # _cm = confusion_matrix(valid_labels, predictions)
        # _cm_metrics: dict[str, float] = {}
        # if _cm.shape == (2, 2):
        #     # Binary classification
        #     TN, FP, FN, TP = _cm.ravel()
        #     _cm_metrics.update({
        #         "TP": TP,
        #         "TN": TN,
        #         "FP": FP,
        #         "FN": FN
        #     })
        # else:
        #     # Multi-class classification
        #     num_classes = _cm.shape[0]
        #     for i in range(num_classes):
        #         TP = _cm[i, i]
        #         FP = _cm[:, i].sum() - TP
        #         FN = _cm[i, :].sum() - TP
        #         TN = _cm.sum() - (TP + FP + FN)
        #         _cm_metrics.update({
        #             f"cls_{i}_TP": TP,
        #             f"cls_{i}_FP": FP,
        #             f"cls_{i}_FN": FN,
        #             f"cls_{i}_TN": TN
        #         })
        #
        # _metrics.update(_cm_metrics)

        if display:
            stars()
            print("Classification Evaluation Metrics")
            lines()
            for key, value in _metrics.items():
                print(f"{key.capitalize():<10}: {value:.4f}")
            stars()
            print()

            stars()
            print("Classification Report")
            lines()
            print(classification_report(valid_labels, predictions))
            stars()
            print()

            stars()
            print("Classification Confusion Matrix")
            lines()
            pprint(confusion_matrix(valid_labels, predictions))
            stars()
            print()
        return _metrics

    def inference(self, sample_feature: DataFrame, sample_label: Series, *, display: bool = False) -> tuple[bool, Any]:
        """
        Inference a single sample prediction.

        :param sample_feature: 2D feature row (e.g. DataFrame.iloc[[row]])
        :param sample_label: The actual label value
        :param display: Whether to display the inference result.
        :return: Tuple of (is_correct, prediction_label)
        """
        pred_label = self.predict(sample_feature)[0]
        status: bool = pred_label == sample_label
        if display:
            print(f"Prediction Result: {'Correct' if status else 'Incorrect'}")
        return status, pred_label


def grid_search_tunes(
        train_features: DataFrame,
        train_labels: Series,
        grid_params: dict[str, list[Any]],
        *,
        mission: str | KNNMissions | Literal["cls", "reg"] = KNNMissions.CLS,
        cv_splits: int = 5,
        cv_shuffle: bool = True,
        randomness: int = 27,
        score_strategy: str | ScoreStrategies | Literal[
            "accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted", "roc_auc_ovr"
        ] = ScoreStrategies.F1_WEIGHTED,
        display: bool = False
) -> GridSearchTonesResponse:
    """
    Perform Grid Search CV to find optimal hyperparameters on training data.

    :param train_features: Training feature matrix.
    :param train_labels: Training label vector.
    :param grid_params: Dictionary with parameters names as keys and lists of parameter settings to try as values.
    :param mission: Type of machine learning task ("cls" for classification, "reg" for regression).
    :param cv_splits: Number of CV folds for tuning (default: 5).
    :param cv_shuffle: Whether to shuffle the training data before splitting (default: True).
    :param randomness: Random state for K-Fold splitting.
    :param score_strategy: Strategy to evaluate the performance on the cross-validated data.
    :param display: Whether to print formatted best parameters and score.
    :return: Dictionary with best_params and best_score.
    """
    base_estimator = KNeighborsClassifier() if mission == "cls" else KNeighborsRegressor()

    if mission == "cls":
        _cv = StratifiedKFold(n_splits=cv_splits, shuffle=cv_shuffle, random_state=randomness)
    else:
        _cv = KFold(n_splits=cv_splits, shuffle=cv_shuffle, random_state=randomness)

    _searcher = GridSearchCV(
        estimator=base_estimator,
        param_grid=grid_params,
        cv=_cv,
        scoring=ScoreStrategies(score_strategy),
        n_jobs=-1
    )
    _searcher.fit(train_features, train_labels)

    best_params: dict = _searcher.best_params_
    best_score: float = float(_searcher.best_score_)

    if display:
        stars()
        print(f"GridSearchCV Hyperparameter Tuning Results ({cv_splits}-Fold CV)")
        lines()
        print(f"Best Scoring Strategy : {score_strategy}")
        print(f"Best CV Score         : {best_score:.4f}")
        print("Best Hyperparameters   :")
        for param, val in best_params.items():
            print(f"- {param:<20}: {val}")
        stars()
        print()

    return GridSearchTonesResponse(
        best_params=best_params,
        best_score=best_score
    )


if __name__ == "__main__":
    pass
