
import numpy
# import scipy

from scipy.stats import norm
from math import sin, cos, exp, log, sqrt
import random

def sample(N, noise_sigma = 0.01, want_sorted_data = False, angle_restriction = 1.0, want_evenly_spaced = False):
    # This updated method is being used to clear some
    # ambiguity concerning the pdf is order to use
    # the KL divergence more easily.
    #
    # When 'angle_restriction' is 1.0 it means
    # that we're not restricting anything.

    if angle_restriction < 0.0 or 1.0 < angle_restriction:
        error("Bad argument for angle_restriction supplied to debian_spiral.sample")

    if not want_evenly_spaced:
        t  = 3 + 12 * numpy.random.random((N,)) * angle_restriction
    else:
        t  = 3 + 12 * numpy.linspace(0,1,N) * angle_restriction

    if want_sorted_data:
        numpy.ndarray.sort(t)

    X = 0.04*t*numpy.sin(t)
    Y = 0.04*t*numpy.cos(t)
    data = numpy.vstack((X,Y)).T

    norms = numpy.sqrt( (data ** 2).sum(axis=1) )


    if noise_sigma > 0.0:
        data = data * ( 1 + numpy.tile( (numpy.random.normal(0, noise_sigma, (N,)) / norms).reshape((-1,1)), (1,2)))

    return data



# Pascal Vincent's definition of the spiral.
# No longer used because it complicated things
# with the definition of the KL divergence.
def pascal_sample(N, noise_sigma = 0.01):
    t  = 3 + 12 * numpy.random.random((N,))
    if noise_sigma > 0.0:
        ex = numpy.random.normal(0, noise_sigma, (N,))
        ey = numpy.random.normal(0, noise_sigma, (N,))
    else:
        ex = numpy.zeros((N,))
        ey = numpy.zeros((N,))

    X = 0.04*t*numpy.sin(t) + ex
    Y = 0.04*t*numpy.cos(t) + ey

    return (X,Y)

