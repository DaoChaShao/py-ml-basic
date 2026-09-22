#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/22 01:01
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_california.py
# @Desc     :   

from pandas import DataFrame, Series
from pprint import pprint

from numpy import ndarray
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch

from utils.ml import (
    get_reg_labels_distribution,
)


def init_california_housing() -> Bunch:
    return fetch_california_housing()


def main() -> None:
    """ Main Function """
    california = init_california_housing()
    features: ndarray = california.data
    labels: ndarray = california.target
    # pprint(features)
    # pprint(labels)

    labels: Series = Series(labels)
    get_reg_labels_distribution(labels, display=True)


if __name__ == "__main__":
    main()
