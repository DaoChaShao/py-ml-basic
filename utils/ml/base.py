#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/20 22:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base.py
# @Desc     :

from abc import ABC, abstractmethod
from enum import StrEnum, unique
from pprint import pprint
from typing import Any, Literal

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from ..highlighter import stars, lines


@unique
class MetricStrategies(StrEnum):
    """
    Metric strategies for multi-class classification metrics.

    - WEIGHTED: Calculate metrics for each label, and find their average weighted by support (sample count).
    - MACRO: Calculate metrics for each label, and find their unweighted mean (does not take label imbalance into account).
    - MICRO: Calculate metrics globally by counting the total true positives, false negatives and false positives.
    - SAMPLES: Calculate metrics for each instance, and find their average (only meaningful for multilabel classification).
    - BINARY: Report metrics for the class specified by pos_label (used for binary classification only).
    """
    WEIGHTED = "weighted"
    MACRO = "macro"
    MICRO = "micro"
    SAMPLES = "samples"
    BINARY = "binary"


class Base(ABC):
    """ Base class for all machine learning models. """

    def __init__(self) -> None:
        """
        Initialize the base class.

        :return: None
        """
        super().__init__()

    @abstractmethod
    def train(self, features: Any, labels: Any) -> None:
        """
        Train the model.

        :param features: The features to train on.
        :param labels: The labels to train on.
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, features: Any) -> Any:
        """
        Predict the labels for the given features.

        :param features: The features to predict on.
        :return: The predicted labels.
        """
        pass

    @staticmethod
    def eval_cls_metrics(
            valid_labels: Any, predictions: Any,
            *,
            strategy: str | MetricStrategies | Literal[
                "weighted", "macro", "micro", "samples", "binary"
            ] = MetricStrategies.WEIGHTED,
            display: bool = False
    ) -> dict[str, float]:
        """
        Evaluate the _acc of the model.

        :param valid_labels: The true labels.
        :param predictions: The predicted labels.
        :param strategy: The average method for multi-class metrics ('weighted', 'macro', etc.).
        :param display: Whether to print the formatted evaluation result.
        :return: Dictionary containing calculated evaluation metrics and matrices.
        """
        _acc: float = float(accuracy_score(valid_labels, predictions))
        _pre: float = float(precision_score(valid_labels, predictions, average=MetricStrategies(strategy)))
        _rec: float = float(recall_score(valid_labels, predictions, average=MetricStrategies(strategy)))
        _f1: float = float(f1_score(valid_labels, predictions, average=MetricStrategies(strategy)))
        _metrics = {
            "accuracy": _acc,
            "precision": _pre,
            "recall": _rec,
            "f1_score": _f1,
        }

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
        return _metrics

    def inference(self, sample_features: Any, sample_label: Any, *, display: bool = False) -> tuple[bool, Any]:
        """
        Inference a single sample prediction.

        :param sample_features: 2D feature row (e.g. DataFrame.iloc[[row]])
        :param sample_label: The actual label value
        :param display: Whether to display the inference result.
        :return: Tuple of (is_correct, prediction_label)
        """
        pred_label = self.predict(sample_features)[0]
        status: bool = pred_label == sample_label
        if display:
            print(f"Prediction Result: {'Correct' if status else 'Incorrect'}")
        return status, pred_label


if __name__ == "__main__":
    pass
