#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/29 21:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   13_time_series_01_arima.py
# @Desc     :   

from pandas import DataFrame, Series, to_datetime, date_range

from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from utils.ml import (
    summary_dataframe,
    get_cls_labels_distribution,
)


def init_bikes() -> Bunch:
    """ Initialize the bikes dataset """
    return fetch_openml(
        "Bike_Sharing_Demand",
        version=2,
        as_frame=True,
    )


def main() -> None:
    """ Main Function """
    bikes: Bunch = init_bikes()
    features: DataFrame = DataFrame(bikes.data, columns=bikes.feature_names)
    labels: Series = Series(bikes.target, name="count")
    print(type(features), type(labels), end="\n\n")

    summary_dataframe(features)

    print(labels, end="\n\n")


if __name__ == "__main__":
    main()
