import numpy as np

from stemu.utils import stack, unstack


def test_stack_unstack():
    N = 10
    k = 5
    d = 20
    X = np.random.rand(N, k)
    t = np.random.rand(d)
    y = np.random.rand(N, d)

    X, y = stack(X, t, y)
    assert X.shape == (N * d, k + 1)
    assert y.shape == (N * d,)

    X, t, y = unstack(X, y)
    assert X.shape == (N, k)
    assert t.shape == (d,)
    assert y.shape == (N, d)
