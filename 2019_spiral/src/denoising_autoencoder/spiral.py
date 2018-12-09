
import numpy as np

from math import sin, cos, exp, log, sqrt
import random


def hyperbolic_sampling(N, a, b, r=None):
    """
    Sample a value t in [a, b] suchs that p(t) propto 1/t.

    You can specify your own values of r in [0,1] if you
    want to use a particular scheme for spacing your points (not recommended).

    Return a vector with shape (N,).

    See my notebook "Research Daily 2018 nov" on 2018-11-30.
    """
    assert 0 < a
    assert a < b
    if a < 1e-16:
        a += 1e-16
        b += 1e-16

    if r is None:
        r = np.random.rand(N)
    else:
        assert np.all(0.0 <= r)
        assert np.all(r <= 1.0)

    x = np.exp( r * np.log(b) + (1.0-r) * np.log(a) )
    return x


def linear_sampling(N, a, b, r=None):
    """
    Sample a value t in [a, b] suchs that p(t) propto t.

    You can specify your own values of r in [0,1] if you
    want to use a particular scheme for spacing your points (not recommended).

    Return a vector with shape (N,).

    See my notebook "Research Daily 2018 nov" on 2018-11-30.
    """
    assert 0 < a
    assert a < b

    if r is None:
        r = np.random.rand(N)
    else:
        assert np.all(0.0 <= r)
        assert np.all(r <= 1.0)

    x = np.sqrt( r * (b*b - a*a) + a*a )
    return x



def sample( N,
            noise_sigma = 0.01,
            want_even_mass_spread = True,
            angle_restriction = 1.0):

    # When 'angle_restriction' is 1.0 it means
    # that we're not restricting anything.
    assert 0.0 <= angle_restriction
    assert angle_restriction <= 1.0

    # It's useful to be able to decide to sample more points as we go further
    # outside the spiral because the density has a tendency to decrease as 1/t
    # simply due to the fact that it covers a larger and larger circle.
    (a, b) = (3, 12 * angle_restriction)
    if want_even_mass_spread:
        # t = hyperbolic_sampling(N, a, b) # no, not that one!
        t = linear_sampling(N, a, b)
    else:
        t  = a + (b-a) * np.random.rand(N)

    X = 0.04 * t * np.sin(t)
    Y = 0.04 * t * np.cos(t)
    data = np.vstack((X,Y)).T

    data = data + noise_sigma * np.random.randn(N, 2)
    return data
