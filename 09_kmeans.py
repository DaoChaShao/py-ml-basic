#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/26 22:28
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   09_kmeans.py
# @Desc     :

from pprint import pprint

from pandas import DataFrame, Series
from sklearn.datasets import load_wine
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import Bunch

from utils.ml import (
    FeaturesRobustScaler,
    KMeansAlgorithms,
    KMeansInitCategories,
    Missions,
    evaluate_kmeans_classification,
    evaluate_kmeans_with_silhouette,
    get_cls_labels_distribution,
    split_data,
    summary_dataframe, evaluate_kmeans_with_ch,
)
from utils.ml.estimators import HyperKMeans


def init_wine() -> Bunch:
    """ Initialize Wine Dataset """
    return load_wine()


def silhouette_scorer(estimator, features):
    """ K-means Scorer """
    return silhouette_score(features, estimator.predict(features))


def ch_scorer(estimator, features):
    """ K-means Scorer """
    return calinski_harabasz_score(features, estimator.predict(features))


def main() -> None:
    """ Main Function """
    wine: Bunch = init_wine()
    features = wine.data
    labels = wine.target
    # print(type(features), type(labels))
    features: DataFrame = DataFrame(features, columns=wine.feature_names)
    labels: Series = Series(labels)
    # print(type(features), type(labels))
    # print(features.head())
    # print(labels.head())

    summary_dataframe(features, display=False)

    get_cls_labels_distribution(labels, display=True)

    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        mission=Missions.CLS,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesRobustScaler(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)

        km = HyperKMeans(3)

        cross_validation: KFold = KFold(n_splits=5, shuffle=True, random_state=27)
        search = GridSearchCV(
            km,
            {
                "n_clusters": [2, 3, 4],
                "init_cat": [KMeansInitCategories.K_MEANS_PLUS_PLUS, KMeansInitCategories.RANDOM],
                "epochs": [300, 500],
                "algorithm": [KMeansAlgorithms.LLOYD, KMeansAlgorithms.ELKAN],
            },
            # scoring=silhouette_scorer,
            scoring=ch_scorer,
            cv=cross_validation,
            n_jobs=-1,  # -1: Use all available CPU cores | 1: Use 1 CPU core
            verbose=1,  # Set to 1 to print whole progress
        )

        search.fit(train_features)
        print("Best Parameters:")
        pprint(search.best_params_)
        print("Best CV Score:", search.best_score_)
        km = search.best_estimator_
        """
        ****************************************************************
        Best Parameters:
        {'algorithm': <KMeansAlgorithms.LLOYD: 'lloyd'>,
         'epochs': 300,
         'init_cat': <KMeansInitCategories.RANDOM: 'random'>,
         'n_clusters': 4}
        ----------------------------------------------------------------
        Best CV Score: 0.2476875972020974
        ****************************************************************
        ****************************************************************
        Best Parameters:
        {'algorithm': <KMeansAlgorithms.LLOYD: 'lloyd'>,
         'epochs': 300,
         'init_cat': <KMeansInitCategories.K_MEANS_PLUS_PLUS: 'k-means++'>,
         'n_clusters': 3}
        ----------------------------------------------------------------
        Best CV Score: 8.874726248755627
        ****************************************************************
        """

        train_predictions = km.predict(train_features)
        evaluate_kmeans_with_ch(train_features, train_predictions, "train", display=True)
        # evaluate_kmeans_with_silhouette(train_features, train_predictions, "train", display=True)
        evaluate_kmeans_classification(train_labels, train_predictions, "train", display=True)
        """
        ****************************************************************
        The function named 'evaluate_kmeans_classification' is starting:
        ----------------------------------------------------------------
        Train ARI      : 0.8991
        Train NMI      : 0.8826
        Train Accuracy : 0.9677
        Train Calinski-Harabasz Score: 41.7377.
        ----------------------------------------------------------------
        The function named 'evaluate_kmeans_classification' took 0.0031 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'evaluate_kmeans_classification' is starting:
        ----------------------------------------------------------------
        Train ARI      : 0.9345
        Train NMI      : 0.9262
        Train Accuracy : 0.9677
        Train Silhouette Score: 0.2683.
        ----------------------------------------------------------------
        The function named 'evaluate_kmeans_classification' took 0.0057 seconds to complete.
        ****************************************************************
        """

        valid_predictions = km.predict(valid_features)
        evaluate_kmeans_with_ch(valid_features, valid_predictions, "valid", display=True)
        # evaluate_kmeans_with_silhouette(valid_features, valid_predictions, "valid", display=True)
        evaluate_kmeans_classification(valid_labels, valid_predictions, "valid", display=True)
        """
        ****************************************************************
        The function named 'evaluate_kmeans_classification' is starting:
        ----------------------------------------------------------------
        Valid ARI      : 0.7640
        Valid NMI      : 0.8215
        Valid Accuracy : 0.9259
        Valid Calinski-Harabasz Score: 11.0807.
        ----------------------------------------------------------------
        The function named 'evaluate_kmeans_classification' took 0.0021 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'evaluate_kmeans_classification' is starting:
        ----------------------------------------------------------------
        Valid ARI      : 0.7640
        Valid NMI      : 0.8215
        Valid Accuracy : 0.9259
        Valid Silhouette Score: 0.2912.
        ----------------------------------------------------------------
        The function named 'evaluate_kmeans_classification' took 0.0018 seconds to complete.
        ****************************************************************
        """

        prove_predictions = km.predict(prove_features)
        evaluate_kmeans_with_ch(prove_features, prove_predictions, "prove", display=True)
        # evaluate_kmeans_with_silhouette(prove_features, prove_predictions, "prove", display=True)
        evaluate_kmeans_classification(prove_labels, prove_predictions, "prove", display=True)
        """
        ****************************************************************
        The function named 'evaluate_kmeans_classification' is starting:
        ----------------------------------------------------------------
        Prove ARI      : 1.0000
        Prove NMI      : 1.0000
        Prove Accuracy : 1.0000
        Prove Calinski-Harabasz Score: 10.1675.
        ----------------------------------------------------------------
        The function named 'evaluate_kmeans_classification' took 0.0018 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'evaluate_kmeans_classification' is starting:
        ----------------------------------------------------------------
        Prove ARI      : 0.9387
        Prove NMI      : 0.9479
        Prove Accuracy : 0.9630
         Prove Silhouette Score: 0.2777.
        ----------------------------------------------------------------
        The function named 'evaluate_kmeans_classification' took 0.0017 seconds to complete.
        ****************************************************************
        """


if __name__ == "__main__":
    main()
