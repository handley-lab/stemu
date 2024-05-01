import numpy as np
from numpy.testing import assert_allclose

from stemu.skutils import CDFTransformer, FunctionScaler, IdentityTransformer


def test_CDFTransformer():
    t = np.linspace(0, 1, 100)
    y = t**3 - t + 1
    y = y * np.random.rand(20, len(t))
    cdf = CDFTransformer()
    assert isinstance(cdf.fit(t, y), CDFTransformer)
    t_ = cdf.transform(t)
    assert not (t == t_).all()
    t_ = cdf.inverse_transform(t_)
    assert (t == t_).all()


def test_FunctionScaler():
    t = np.linspace(0, 1, 100)
    y = t**3 - t + 1
    y = y * np.random.rand(20, len(t))
    X = np.block([[t], [y]])

    fs = FunctionScaler()
    assert isinstance(fs.fit(X), FunctionScaler)
    X_ = fs.transform(X)
    assert_allclose(X_[1:].mean(axis=0), 0, atol=1e-15)
    assert_allclose(X_[1:].std(axis=0), 1, atol=1e-15)
    assert_allclose(fs.inverse_transform(X_), X)


def test_IdentityTransformer():
    y = np.random.rand(10, 20)

    identity = IdentityTransformer()
    assert isinstance(identity.fit(y), IdentityTransformer)
    y_ = identity.transform(y)
    assert (y == y_).all()
    y_ = identity.inverse_transform(y_)
    assert (y == y_).all()
