#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/27 23:41
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   10_gmm_03_lda.py
# @Desc     :

from pprint import pprint

from pandas import DataFrame, Series
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import Bunch

from utils.ml import (
    FeaturesCategories,
    FeaturesRobustScaler,
    GMMCovarianceCategories,
    GMMInitParamsCategories,
    LDAAssessor,
    Missions,
    eval_clustering_classification,
    eval_clustering_with_ch,
    get_cls_labels_distribution,
    split_data,
    summary_dataframe,
)
from utils.ml.estimators import HyperGaussianMixture


def init_wine() -> Bunch:
    """ Initialize Wine Dataset """
    return load_wine()


def aic_scorer(estimator, features):
    """ GMM AIC Scorer """
    return -estimator.aic(features)


def bic_scorer(estimator, features):
    """ GMM BIC Scorer """
    return -estimator.bic(features)


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

    _, _, _, _, n_classes = get_cls_labels_distribution(labels, display=True)

    train_features, valid_features, prove_features, train_labels, valid_labels, prove_labels = split_data(
        features, labels,
        mission=Missions.CLS,
        randomness=27, shuffle_status=True, display=True
    )

    with FeaturesRobustScaler(train_features) as standardiser:
        train_features = standardiser.transform(train_features)
        valid_features = standardiser.transform(valid_features)
        prove_features = standardiser.transform(prove_features)

        # LDA
        with LDAAssessor(features=train_features, labels=train_labels, n_components=n_classes) as assessor:
            train_features = assessor.reduce(train_features, FeaturesCategories.TRAIN, display=True)
            valid_features = assessor.reduce(valid_features, FeaturesCategories.VALID, display=True)
            prove_features = assessor.reduce(prove_features, FeaturesCategories.PROVE, display=True)

        gmm = HyperGaussianMixture(3)

        # GridSearchCV
        cross_validation: KFold = KFold(n_splits=5, shuffle=True, random_state=27)
        search = GridSearchCV(
            gmm,
            {
                "n_components": [2, 3, 4, 5],
                "covariance": [
                    GMMCovarianceCategories.FULL,
                    GMMCovarianceCategories.DIAG,
                    GMMCovarianceCategories.SPHERICAL,
                    GMMCovarianceCategories.TIED,
                ],
                "epochs": [100, 300],
                "init_params": [
                    GMMInitParamsCategories.KMEANS,
                    GMMInitParamsCategories.K_MEANS_PLUS_PLUS,
                    GMMInitParamsCategories.RANDOM,
                    GMMInitParamsCategories.RANDOM_FROM_DATA,
                ],
            },
            scoring=aic_scorer,
            # scoring=bic_scorer,
            cv=cross_validation,
            n_jobs=-1,  # -1: Use all available CPU cores | 1: Use 1 CPU core
            verbose=1,  # Set to 1 to print whole progress
        )

        search.fit(train_features)
        print("Best Parameters:")
        pprint(search.best_params_)
        print("Best CV Score:", search.best_score_)
        best_n_components = search.best_params_["n_components"]
        print("Best n_components:", best_n_components)
        """
        ****************************************************************
        Best Parameters:
        {'covariance': <GMMCovarianceCategories.TIED: 'tied'>,
         'epochs': 100,
         'init_params': <GMMInitParamsCategories.KMEANS: 'kmeans'>,
         'n_components': 4}
        ----------------------------------------------------------------
        Best CV Score: -10.548456325032957
        ----------------------------------------------------------------
        Best n_components: 4
        ****************************************************************
        """

        gmm = search.best_estimator_
        train_predictions = gmm.predict(train_features)
        eval_clustering_with_ch(train_features, train_predictions, "train", display=True)
        # evaluate_kmeans_with_silhouette(train_features, train_predictions, "train", display=True)
        eval_clustering_classification(train_labels, train_predictions, "train", display=True)
        """
        ****************************************************************
        The function named 'eval_clustering_classification' is starting:
        ----------------------------------------------------------------
        Train ARI      : 0.7968
        Train NMI      : 0.8490
        Train Accuracy : 0.8548
        ----------------------------------------------------------------
        Train Calinski-Harabasz Score: 32.4531.
        ----------------------------------------------------------------
        The function named 'eval_clustering_classification' took 0.0052 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'eval_clustering_classification' is starting (After PCA):
        ----------------------------------------------------------------
        Train ARI      : 0.1114
        Train NMI      : 0.2382
        Train Accuracy : 0.5161
        ----------------------------------------------------------------
        Train Calinski-Harabasz Score: 19.0711.
        ----------------------------------------------------------------
        The function named 'eval_clustering_classification' took 0.0039 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'eval_clustering_classification' is starting (After LDA):
        ----------------------------------------------------------------
        Train ARI      : 0.9759
        Train NMI      : 0.9650
        Train Accuracy : 0.9919
        ----------------------------------------------------------------
        Train Calinski-Harabasz Score: 413.3599.
        ----------------------------------------------------------------
        The function named 'eval_clustering_classification' took 0.0044 seconds to complete.
        ****************************************************************
        """

        valid_predictions = gmm.predict(valid_features)
        eval_clustering_with_ch(valid_features, valid_predictions, "valid", display=True)
        # evaluate_kmeans_with_silhouette(valid_features, valid_predictions, "valid", display=True)
        eval_clustering_classification(valid_labels, valid_predictions, "valid", display=True)
        """
        ****************************************************************
        The function named 'eval_clustering_classification' is starting:
        ----------------------------------------------------------------
        Valid ARI      : 0.6926
        Valid NMI      : 0.7933
        Valid Accuracy : 0.7778
        ----------------------------------------------------------------
        Valid Calinski-Harabasz Score: 7.8394.
        ----------------------------------------------------------------
        The function named 'eval_clustering_classification' took 0.0020 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'eval_clustering_classification' is starting (After PCA):
        ----------------------------------------------------------------
        Valid ARI      : 0.1687
        Valid NMI      : 0.3229
        Valid Accuracy : 0.5556
        ----------------------------------------------------------------
        Valid Calinski-Harabasz Score: 5.0432.
        ----------------------------------------------------------------
        The function named 'eval_clustering_classification' took 0.0026 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'eval_clustering_classification' is starting (After LDA):
        ----------------------------------------------------------------
        Valid ARI      : 0.8751
        Valid NMI      : 0.8872
        Valid Accuracy : 0.9630
        ----------------------------------------------------------------
        Valid Calinski-Harabasz Score: 69.8099.
        ----------------------------------------------------------------
        The function named 'eval_clustering_classification' took 0.0018 seconds to complete.
        ****************************************************************
        """

        prove_predictions = gmm.predict(prove_features)
        eval_clustering_with_ch(prove_features, prove_predictions, "prove", display=True)
        # evaluate_kmeans_with_silhouette(prove_features, prove_predictions, "prove", display=True)
        eval_clustering_classification(prove_labels, prove_predictions, "prove", display=True)
        """
        ****************************************************************
        The function named 'eval_clustering_classification' is starting:
        ----------------------------------------------------------------
        Prove ARI      : 0.8891
        Prove NMI      : 0.9219
        Prove Accuracy : 0.9259
        ----------------------------------------------------------------
        Prove Calinski-Harabasz Score: 8.5937.
        ----------------------------------------------------------------
        The function named 'eval_clustering_classification' took 0.0018 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'eval_clustering_classification' is starting (After PCA):
        ----------------------------------------------------------------
        Prove ARI      : 0.0685
        Prove NMI      : 0.2118
        Prove Accuracy : 0.4815
        ----------------------------------------------------------------
        Prove Calinski-Harabasz Score: 2.9824.
        ----------------------------------------------------------------
        The function named 'eval_clustering_classification' took 0.0018 seconds to complete.
        ****************************************************************
        ****************************************************************
        The function named 'eval_clustering_classification' is starting (After LDA):
        ----------------------------------------------------------------
        Prove ARI      : 1.0000
        Prove NMI      : 1.0000
        Prove Accuracy : 1.0000
        ----------------------------------------------------------------
        Prove Calinski-Harabasz Score: 102.8395.
        ----------------------------------------------------------------
        The function named 'eval_clustering_classification' took 0.0017 seconds to complete.
        ****************************************************************
        """


if __name__ == "__main__":
    main()
