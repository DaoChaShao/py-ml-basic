#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 01:04
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   postprocessor.py
# @Desc     :

from utils import green, red, timer, yellow


@timer
def diagnose_reg_fit(
        train_rmse: float,
        valid_rmse: float,
        train_labels_std: float,
        *,
        overfit_threshold: float = 0.15,
        underfit_threshold: float = 0.80,
        display: bool = False,
) -> str:
    """
    Diagnose if the model is underfitting, overfitting, or well-fitted.

    :param train_rmse: Training set RMSE.
    :param valid_rmse: Validation set RMSE.
    :param train_labels_std: Standard deviation of the training set labels (for scale invariance).
    :param overfit_threshold: Percentage difference threshold between valid and train error to detect overfitting.
    :param underfit_threshold: Relative threshold (Train RMSE / Label Std) to detect underfitting.
    :param display: Whether to display the diagnosis result.
    :return: Diagnosis result string.
    """
    _overfit_ratio: float = (valid_rmse - train_rmse) / train_rmse
    _underfit_ratio: float = train_rmse / train_labels_std

    if _overfit_ratio > overfit_threshold:
        _diagnosis: str = red("Overfitting Detected! (Valid error is {:.2%} higher than Train)")
    elif _underfit_ratio > underfit_threshold:
        _diagnosis: str = yellow("Underfitting Detected! (Train error is {:.2%} of label std)")
    else:
        _diagnosis: str = green("Well-Fitted! (Good generalization performance)")

    if display:
        print(f"Model Diagnosis: {_diagnosis}")
    return _diagnosis


@timer
def diagnose_cls_fit(
        train_f1_score: float,
        valid_f1_score: float,
        *,
        overfit_threshold: float = 0.08,
        underfit_threshold: float = 0.75,
        display: bool = False,
) -> str:
    """
    Diagnose if the classification model is underfitting, overfitting, or well-fitted.

    :param train_f1_score: Training set F1 score (or Accuracy).
    :param valid_f1_score: Validation set F1 score (or Accuracy).
    :param overfit_threshold: Drop in performance from train to valid to detect overfitting.
    :param underfit_threshold: Minimum acceptable baseline metric on train set to avoid underfitting.
    :param display: Whether to display the diagnosis result.
    :return: Diagnosis result string.
    """
    _f1_drop: float = train_f1_score - valid_f1_score

    if train_f1_score < underfit_threshold:
        _diagnosis: str = yellow(f"Underfitting Detected! (Train F1 is low: {train_f1_score:.2%})")
    elif _f1_drop > overfit_threshold:
        _diagnosis: str = red(f"Overfitting Detected! (Valid F1 is {_f1_drop:.2%} lower than Train)")
    else:
        _diagnosis: str = green("Well-Fitted! (Good generalization performance)")

    if display:
        print(f"Model Diagnosis: {_diagnosis}")
    return _diagnosis


if __name__ == "__main__":
    pass
