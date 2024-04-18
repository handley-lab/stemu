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
        return self.cdf(X)

    def inverse_transform(self, X):
        return self.icdf(X)

    def fit(self, X, y=None):
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
        t, y = X[0], X[1:]
        y = (y - self.mean(t)) / self.std(t)
        return np.block([[t], [y]])

    def inverse_transform(self, X):
        t, y = X[0], X[1:]
        y = y * self.std(t) + self.mean(t)
        return np.block([[t], [y]])

    def fit(self, X, y=None):
        t, y = X[0], X[1:]
        self.mean = interp1d(t, y.mean(axis=0))
        self.std = interp1d(t, y.std(axis=0))
        return self


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X
