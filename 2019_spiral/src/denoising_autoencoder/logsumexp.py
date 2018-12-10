
import numpy as np
import numpy.testing as npt


def logsumexp(A, axis):
    """
    A contains the values equivalent to A = np.log(V).
    We want to compute np.log(V.sum(axis=axis)) but in a numerically-stable way.

    If A is of shape (K, K, N), this returns an array of shape (K, K).
    """
    M = np.max(A, axis=axis, keepdims=True)
    B = A - M

    #lower_bound = -500
    #B[B < lower_bound] = lower_bound
    #assert B.max() <= 1.0
    #assert lower_bound <= B.min()

    r = np.log(np.exp(B).sum(axis=axis)) + M.squeeze()
    return r


def test_logsumexp_01():

    shape = (4, 5, 6)
    V = np.ones(shape)
    A = np.log(V)

    for axis in range(len(shape)):
        ground_truth = np.log(V.sum(axis=axis))
        result = logsumexp(A, axis)
        npt.assert_allclose(result, ground_truth)


def test_logsumexp_02():

    shape = (4, 5, 6)
    V = 10*np.random.rand(*shape) + 1e-8
    A = np.log(V)

    for axis in range(len(shape)):
        ground_truth = np.log(V.sum(axis=axis))
        result = logsumexp(A, axis)
        npt.assert_allclose(result, ground_truth)
