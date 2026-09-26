#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/23 14:37
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   logistic.py
# @Desc     :

from typing import Any, Literal, override, Self

from access_modifiers import protectedmethod
from pandas import DataFrame, Series
from pydantic import Field, validate_call
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from ..types import LogisticSolvers
from .base import Base


class LogisticRegClassifier(Base):
    """ Logistic Regression Classifier Wrapper. """

    @validate_call
    def __init__(
            self,
            *,
            penalty_strength: float = Field(1.0, gt=0, description="Lower strength (C), stronger regularisation."),
            l1_ratio: float = Field(0.5, ge=0, le=1, description="l1_ratio = 1 is Lasso, l1_ratio = 0 is Ridge."),
            is_intercept: bool = True,
            randomness: int = 27,
            solver: str | LogisticSolvers | Literal[
                "lbfgs", "sag", "saga", "liblinear", "newton-cg", "newton-cholesky"
            ] = LogisticSolvers.SAGA,
            epochs: int = Field(100, gt=0, description="Number of iterations to run gradient descent for."),

    ) -> None:
        """
        Initialise the Logistic Regression Classifier.

        :param penalty_strength: Inverse of regularisation strength (C).
        :param l1_ratio: ElasticNet mixing parameter (0 for L2, 1 for L1).
        :param is_intercept: Whether to calculate the intercept for this model.
        :param randomness: The random seed for reproducibility.
        :param solver: The solver to use for optimisation.
        :param epochs: The maximum number of iterations for solver convergence.
        :return: None
        """
        super().__init__()
        self._strength: float = penalty_strength
        self._ratio: float = l1_ratio
        self._is_intercept: bool = is_intercept
        self._randomness: int = randomness
        self._solver: LogisticSolvers = LogisticSolvers(solver)
        self._epochs: int = epochs

    @protectedmethod
    def _init_model(self) -> None:
        """
       Initialise the underlying Scikit-Learn LogisticRegression estimator.

       :return: None
       """
        self._model = LogisticRegression(
            C=self._strength,
            l1_ratio=self._ratio,
            fit_intercept=self._is_intercept,
            random_state=self._randomness,
            solver=self._solver.value,
            max_iter=self._epochs,
            tol=1e-3,
        )

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
        Train the Logistic Regression Classifier with the given features and labels.

        :param features: The features (independent variables) for training.
        :param labels: The labels (dependent variable) for training.
        :return: The estimator
        """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")

        self._model.fit(features, labels)
        self._fitted = True
        return self

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict the target labels for the given features using the trained model.

        :param features: The features (independent variables) for prediction.
        :return: The predicted target labels.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    @property
    def C(self) -> float:
        """
        Get the regularisation strength C (alpha).

        :return: The alpha value.
        """
        return self._strength

    @property
    def l1_ratio(self) -> float:
        """
        Get the l1 ratio.

        :return: The l1 ratio value.
        """
        return self._ratio

    @property
    def coefficient(self) -> Any:
        """
        Get the regression coefficients (weights).

        :return: The regression coefficients.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.coef_

    @property
    def intercept(self) -> Any:
        """
        Get the regression intercept (bias).

        :return: The regression intercept value or None if disabled.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        if not self._is_intercept:
            return None
        return self._model.intercept_

    @property
    def solver(self) -> LogisticSolvers:
        """
        Get the solver used for optimisation.

        :return: The solver type.
        """
        return self._solver

    def __repr__(self) -> str:
        """
        Get the string representation of the estimator.

        :return: The string representation.
        """
        return (
            f"LogisticRegClassifier("
            f"C(penalty_strength)={self._strength}, "
            f"l1_ratio={self._ratio}, "
            f"fit_intercept={self._is_intercept}, "
            f"random_state={self._randomness}, "
            f"solver={self._solver}, "
            f"max_iter={self._epochs})"
        )


class HyperLogisticRegClassifier(Base, BaseEstimator):
    """ Logistic Regression Classifier Wrapper. """

    @validate_call
    def __init__(
            self,
            *,
            penalty_strength: float = Field(1.0, gt=0, description="Lower strength (C), stronger regularisation."),
            l1_ratio: float = Field(0.5, ge=0, le=1, description="l1_ratio = 1 is Lasso, l1_ratio = 0 is Ridge."),
            is_intercept: bool = True,
            randomness: int = 27,
            solver: str | LogisticSolvers | Literal[
                "lbfgs", "sag", "saga", "liblinear", "newton-cg", "newton-cholesky"
            ] = LogisticSolvers.SAGA,
            epochs: int = Field(100, gt=0, description="Number of iterations to run gradient descent for."),

    ) -> None:
        """
        Initialise the Logistic Regression Classifier.

        :param penalty_strength: Inverse of regularisation strength (C).
        :param l1_ratio: ElasticNet mixing parameter (0 for L2, 1 for L1).
        :param is_intercept: Whether to calculate the intercept for this model.
        :param randomness: The random seed for reproducibility.
        :param solver: The solver to use for optimisation.
        :param epochs: The maximum number of iterations for solver convergence.
        :return: None
        """
        super().__init__()
        self.strength: float = penalty_strength
        self.ratio: float = l1_ratio
        self.is_intercept: bool = is_intercept
        self.randomness: int = randomness
        self.solver: LogisticSolvers = LogisticSolvers(solver)
        self.epochs: int = epochs

    @protectedmethod
    def _init_model(self) -> None:
        """
       Initialise the underlying Scikit-Learn LogisticRegression estimator.

       :return: None
       """
        self._model = LogisticRegression(
            C=self.strength,
            l1_ratio=self.ratio,
            fit_intercept=self.is_intercept,
            random_state=self.randomness,
            solver=self.solver.value,
            max_iter=self.epochs,
            tol=1e-3,
        )

    @override
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        This method allows sklearn utilities such as GridSearchCV to inspect and clone the estimator.

        :param deep: Whether to return parameters of nested estimators.
        :return: Estimator parameters.
        """
        return {
            "penalty_strength": self.strength,
            "l1_ratio": self.ratio,
            "is_intercept": self.is_intercept,
            "randomness": self.randomness,
            "solver": self.solver,
            "epochs": self.epochs,
        }

    @override
    def set_params(self, **params: Any) -> Self:
        """
        This method is required by sklearn's hyperparameter search utilities.

        :param params: Parameters to set.
        :return: The estimator with parameters set.
        """
        if not params:
            return self

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for HyperLogisticRegClassifier. "
                    f"Valid parameters are: {list(valid_params)}."
                )
            setattr(self, key, value)

        self._fitted = False
        return self

    @override
    def fit(self, features: DataFrame, labels: Series) -> Self:
        """
        Train the Logistic Regression Classifier with the given features and labels.

        :param features: The features (independent variables) for training.
        :param labels: The labels (dependent variable) for training.
        :return: The estimator
        """
        self._init_model()

        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")

        self._model.fit(features, labels)
        self._fitted = True
        return self

    @override
    def predict(self, features: DataFrame) -> Any:
        """
        Predict the target labels for the given features using the trained model.

        :param features: The features (independent variables) for prediction.
        :return: The predicted target labels.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.predict(features)

    @property
    def C(self) -> float:
        """
        Get the regularisation strength C (alpha).

        :return: The alpha value.
        """
        return self.strength

    @property
    def l1_ratio(self) -> float:
        """
        Get the l1 ratio.

        :return: The l1 ratio value.
        """
        return self.ratio

    @property
    def coefficient(self) -> Any:
        """
        Get the regression coefficients (weights).

        :return: The regression coefficients.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        return self._model.coef_

    @property
    def intercept(self) -> Any:
        """
        Get the regression intercept (bias).

        :return: The regression intercept value or None if disabled.
        """
        if self._model is None:
            raise RuntimeError("Estimator has not been initialised.")
        if not self._fitted:
            raise RuntimeError("Estimator has not been trained yet. Call `train()` first.")
        if not self.is_intercept:
            return None
        return self._model.intercept_
