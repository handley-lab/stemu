"""Utilities for working with scikit-learn."""

import numpy as np
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class CDFTransformer(BaseEstimator, TransformerMixin):
    """Transform independent variable using CDF from dependent variable.

    The CDF is defined by the cumulative sum of the standard deviation of the
    dependent variable.

    This is in the style of other sklearn transformers.
    """

    def transform(self, X):
        """Transform the independent variable.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        """
        return self.cdf(X)

    def inverse_transform(self, X):
        """Inverse transform the independent variable.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        """
        return self.icdf(X)

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples, n_target)
            The dependent variable for the target
        """
        cdf = y.std(axis=0).cumsum() / y.std(axis=0).sum()
        self.cdf = interp1d(X, cdf)
        self.icdf = interp1d(cdf, X)
        return self


class FunctionScaler(BaseEstimator, TransformerMixin):
    """Scale dependent variable.

    The function is defined by the mean and standard deviation of the dependent
    variable (as a function of the independent variable).
    """

    def transform(self, X):
        """Transform the dependent variable.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        """
        t, y = X[0], X[1:]
        y = (y - self.mean(t)) / self.std(t)
        return np.block([[t], [y]])

    def inverse_transform(self, X):
        """Inverse transform the dependent variable.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        """
        t, y = X[0], X[1:]
        y = y * self.std(t) + self.mean(t)
        return np.block([[t], [y]])

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples, n_target)
            The dependent variable for the target
        """
        t, y = X[0], X[1:]
        self.mean = interp1d(t, y.mean(axis=0))
        self.std = interp1d(t, y.std(axis=0))
        return self


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Do nothing transformer."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        """
        return self

    def transform(self, X, y=None):
        """Transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples, n_target)
            The dependent variable for the target
        """
        return X

    def inverse_transform(self, X):
        """Inverse transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        """
        return X
