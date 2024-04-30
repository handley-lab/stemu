"""Emulator base class."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from tensorflow import keras

from stemu.skutils import CDFTransformer, FunctionScaler, IdentityTransformer
from stemu.utils import stack, unstack

default_network = [
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
]


class Emu(object):
    """General Emulation base class.

    This fits an emulator for y=f(t|X) in the style of sklearn models.

    Anything with a default initialisation in the __init__ method is considered
    a hyperparameter and can be adjusted by the user after initialisation.

    Attributes
    ----------
    model : keras model, default is a simple dense network
    epochs : int, default=100
    loss : keras loss, default='mse'
    optimizer : keras optimizer, default='adam'
    callbacks : list of keras.callbacks
    X_pipeline : sklearn.pipeline to transform input data X
    t_pipeline : sklearn.pipeline to transform independent variable t
    y_pipeline : sklearn.pipeline to transform dependent variable y
    ty_pipeline : sklearn.pipeline to transform independent and dependent
                  variables simultaneously
    """

    def __init__(self, *args, **kwargs):
        self.epochs = 100
        self.loss = "mse"
        self.optimizer = "adam"
        self.callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=3)]

        self.X_pipeline = Pipeline([("scaler", StandardScaler())])
        self.t_pipeline = Pipeline([("cdf", CDFTransformer())])
        self.y_pipeline = Pipeline([("default", IdentityTransformer())])
        self.ty_pipeline = Pipeline([("scaler", FunctionScaler())])

    def fit(self, X, t, y):
        """Fit the emulator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        t : array-like of shape (n_target,)
            The independent variable for the target
        y : array-like of shape (n_samples, n_target)
            The dependent variable for the target

        Returns
        -------
        self : object
            Returns self.
        """
        self.t = t

        X = self.X_pipeline.fit_transform(X)
        y = self.y_pipeline.fit_transform(y)
        t = self.t_pipeline.fit_transform(t, y)

        ty = self.ty_pipeline.fit_transform(np.block([[t], [y]]))
        t, y = ty[0], ty[1:]

        X, y = stack(X, t, y)

        self.model = keras.models.Sequential(
            [keras.layers.Input(X.shape[-1:])]
            + default_network
            + [keras.layers.Dense(1, activation="linear")]
        )

        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.history = self.model.fit(
            X, y, epochs=self.epochs, batch_size=len(t), callbacks=self.callbacks
        )
        return self

    def predict(self, X, t=None):
        """Predict the target.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        t : array-like of shape (n_target,)
            The independent variable for the target
            Defaults to the original training t

        Returns
        -------
        y : array-like of shape (n_samples, n_target)
            The predicted target
        """
        if t is None:
            t = self.t
        t = self.t_pipeline.transform(t)
        X = self.X_pipeline.transform(np.atleast_2d(X))
        X, _ = stack(X, np.atleast_1d(t), 1)
        y = self.model.predict(X)
        _, _, y = unstack(X, y)
        ty = self.ty_pipeline.inverse_transform(np.block([[t], [y]]))
        _, y = ty[0], ty[1:]
        y = self.y_pipeline.inverse_transform(y)
        return y
