#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/20 22:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base.py
# @Desc     :

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, Self

from joblib import dump, load
from pandas import DataFrame, Series
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from utils.helper import Access
from utils.highlighter import lines, stars
from utils.ml.types import AveStrategies, Missions


class Base(ABC, Access):
    """ Base class for all machine learning models. """

    def __init__(self) -> None:
        """
        Initialize the base class.

        :return: None
        """
        super().__init__()
        self._model: Any = None
        self._fitted: bool = False

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
            print(classification_report(valid_labels, predictions, digits=4))
            stars()
            print()

            stars()
            print("Classification Confusion Matrix")
            lines()
            pprint(confusion_matrix(valid_labels, predictions))
            stars()
            print()
        return _metrics

    @staticmethod
    def eval_reg(
            valid_labels: Series, predictions: Series,
            *,
            display: bool = False
    ) -> dict[str, Any]:
        """
        Evaluate the regression performance of the model.

        :param valid_labels: The true continuous target values.
        :param predictions: The predicted continuous target values.
        :param display: Whether to print the formatted evaluation result.
        :return: Dictionary containing calculated evaluation metrics.
        """
        _mse: float = mean_squared_error(valid_labels, predictions)
        _rmse: float = root_mean_squared_error(valid_labels, predictions)
        _mae: float = mean_absolute_error(valid_labels, predictions)
        _r2: float = r2_score(valid_labels, predictions)
        _mape: float = mean_absolute_percentage_error(valid_labels, predictions)

        _metrics = {
            "mse": _mse,
            "rmse": _rmse,
            "mae": _mae,
            "r2": _r2,
            "mape": _mape
        }

        if display:
            stars()
            print("Regression Evaluation Metrics")
            lines()
            print(f"R² Score  : {_r2:.4f}")
            print(f"RMSE      : {_rmse:.4f}")
            print(f"MAE       : {_mae:.4f}")
            print(f"MSE       : {_mse:.4f}")
            print(f"MAPE      : {_mape:.4%}")
            stars()
            print()

        return _metrics

    def inference(
            self,
            sample_feature: DataFrame | Series, sample_label: Series,
            *,
            mission: str | Missions | Literal["cls", "reg"] = Missions.CLS,
            error_thresholds: tuple[float, float] = (0.0, 1.0),
            display: bool = False
    ) -> tuple[bool, Any]:
        """
        Inference a single sample prediction.

        :param sample_feature: 2D feature row (e.g. DataFrame.iloc[[row]])
        :param sample_label: The actual label value
        :param mission: Type of machine learning task ("cls" for classification, "reg" for regression).
        :param error_thresholds: Error bounds (excellent_threshold, acceptable_threshold) used to rate prediction quality. 1 - 2 RMSE units
        :param display: Whether to display the inference result.
        :return: Tuple of (is_correct, prediction_label)
        """
        if isinstance(sample_feature, Series):
            sample_feature = sample_feature.to_frame().T

        true_label = sample_label.iloc[0] if isinstance(sample_label, Series) else sample_label
        pred_label = self.predict(sample_feature)[0]

        match Missions(mission):
            case Missions.CLS:
                status: bool = pred_label == true_label
                if display:
                    print(f"Prediction Result: {'Correct' if status else 'Incorrect'}")
                return status, pred_label

            case Missions.REG:
                error: float = abs(true_label - pred_label)
                status: bool = error <= error_thresholds[1]

                if display:
                    if error <= error_thresholds[0]:
                        level = "Excellent"
                    elif error <= error_thresholds[1]:
                        level = "Acceptable"
                    else:
                        level = "Unreasonable"
                    print(f"Pred: {pred_label:.4f} | True: {true_label:.4f} | Error: {error:.4f} | Level: {level}")
                return status, pred_label
            case _:
                raise TypeError("There is no such mission!")

    def save(
            self,
            *,
            model_dir: str | Path = "models",
            model_name: str | Path = "model",
            display: bool = True
    ) -> None:
        """
        Save the model to a file.

        :param model_dir: The directory to save the model.
        :param model_name: The name of the model file.
        :param display: Whether to display the save path.
        :return: None
        """
        _target_dir = Path(model_dir).resolve()
        # Create a new dir if it does not exist
        _target_dir.mkdir(parents=True, exist_ok=True)

        _pure_name = Path(model_name).stem
        _timer = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        save_path = _target_dir / f"trained_at_{_timer}_{_pure_name}.pt"
        dump(self, save_path)

        if display:
            print(f"Model successfully saved to: {save_path}")

    @classmethod
    def load(cls, model_path: str | Path) -> Self:
        """
        Load the model from a file.

        :param model_path: The path to the model file.
        :return: The loaded model.
        """
        _path = Path(model_path).resolve()
        if not _path.exists():
            raise FileNotFoundError(f"Model file not found at: {_path}")
        return load(_path)


if __name__ == "__main__":
    pass
