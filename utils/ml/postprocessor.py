#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 01:04
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   postprocessor.py
# @Desc     :

from utils import green, red, timer, yellow


@timer
def diagnose_fit(
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
    overfit_ratio: float = (valid_rmse - train_rmse) / train_rmse
    underfit_ratio: float = train_rmse / train_labels_std

    if overfit_ratio > overfit_threshold:
        diagnosis: str = red("Overfitting Detected! (Valid error is {:.2%} higher than Train)")
    elif underfit_ratio > underfit_threshold:
        diagnosis: str = yellow("Underfitting Detected! (Train error is {:.2%} of label std)")
    else:
        diagnosis: str = green("Well-Fitted! (Good generalization performance)")

    if display:
        print(f"Model Diagnosis: {diagnosis}")
    return diagnosis


if __name__ == "__main__":
    pass
