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
        valid_r2: float,
        train_labels_std: float,
        *,
        overfit_threshold: float = 0.15,
        underfit_threshold: float = 0.80,
        display: bool = False,
) -> str:
    """
    Diagnose regression model fitting status.

    :param train_rmse: Training set RMSE.
    :param valid_rmse: Validation set RMSE.
    :param valid_r2: Validation set R² score.
    :param train_labels_std: Standard deviation of training set labels.
    :param overfit_threshold: Maximum acceptable generalisation gap.
    :param underfit_threshold: Maximum acceptable train RMSE relative to label std.
    :param display: Whether to display the diagnosis result.
    :return: Diagnosis result string.
    """
    _generalization_gap: float = (valid_rmse - train_rmse) / train_rmse
    _train_error_ratio: float = train_rmse / train_labels_std

    if _train_error_ratio > underfit_threshold:
        _diagnosis = yellow(f"Underfitting Detected! (Train error is {_train_error_ratio:.2%} of label std)")
    elif _generalization_gap > overfit_threshold:
        _diagnosis = red(f"Overfitting Detected! (Valid error is {_generalization_gap:.2%} higher than Train)")
    elif valid_r2 < 0.0:
        _diagnosis = yellow(f"Poor Generalization! (Validation R² = {valid_r2:.4f})")
    else:
        _diagnosis = green(f"Well-Fitted! (Validation R² = {valid_r2:.4f})")

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
