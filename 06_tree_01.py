#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 22:09
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_tree_01.py
# @Desc     :   

from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from utils.ml import (
    get_cls_labels_distribution,
    summary_dataframe,
)


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
    # print(features.head())
    # print(labels.head())
    print(type(features), type(labels))

    summary_dataframe(features)

    get_cls_labels_distribution(labels, display=True)


if __name__ == "__main__":
    main()
