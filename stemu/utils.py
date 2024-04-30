"""Utility functions for stemu."""

import numpy as np
import pandas as pd


def stack(X, t, y):
    """Stack the data for training.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
    t : array-like of shape (n_target)
        The independent variable for the target
    y : array-like of shape (n_samples, n_target)
        The dependent variable for the target

    Returns
    -------
    X : array-like of shape (n_samples*n_target,n_features)
        The input data.
    y : array-like of shape (n_samples*n_target,)
        The dependent variable for the target
    """
    data = pd.DataFrame(
        y, columns=t, index=pd.MultiIndex.from_arrays(np.atleast_2d(X).T)
    ).stack()
    y = data.to_numpy()
    X = data.index.to_frame().to_numpy()
    return X, y


def unstack(X, y):
    """Unstack the data for prediction.

    Parameters
    ----------
    X : array-like of shape (n_samples*n_target, n_features)
        The input data.
    y : array-like of shape (n_samples*n_target,)
        The dependent variable for the target

    Returns
    -------
    X : array-like of shape (n_samples, n_features)
        The input data.
    t : array-like of shape (n_target,)
        The independent variable for the target
    y : array-like of shape (n_samples, n_target)
        The dependent variable for the target
    """
    data = pd.DataFrame(y, index=pd.MultiIndex.from_arrays(np.atleast_2d(X).T)).unstack(
        sort=False
    )
    y = data.to_numpy()
    X = data.index.to_frame().to_numpy()
    t = data.columns.get_level_values(1).to_numpy()
    return X, t, y
