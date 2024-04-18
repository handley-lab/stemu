import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.model_selection import train_test_split

from stemu.emu import Emu


def test_emu():
    # Polynomial test case
    N = 100
    d = 10
    k = 3
    t = np.linspace(-1, 1, d)
    X = np.random.randn(N, k)
    i = np.arange(k)
    y = X @ t ** i[:, None]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    emu = Emu()
    emu.epochs = 1000
    emu.fit(X_train, t, y_train)

    train = (y_train - emu.predict(X_train)).flatten()
    test = (y_test - emu.predict(X_test)).flatten()
    assert test.std() < train.std() * 5
