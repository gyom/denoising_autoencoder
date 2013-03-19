
############################################################

# This turns out to be more assoc than conj, but
# it was just a prototype implementation for fun.
# We'll see when it comes to mimicking actual clojure.
def conj(E, o):
    if type(E) == list:
        return E + [o]
    elif type(E) == dict:
        return dict(E.items() + [o])
    else:
        #dict(E.items() + [o])
        raise BaseException("Unsure how to process the arguments given to conj. type was %s" % str(type(E)),)


def get_dict_key_or_default(D, key, default = None, want_error_if_missing = False):
    if D.has_key(key):
        return D[key]
    else:
        if not want_error_if_missing:
            return default
        else:
            raise("Cannot find key %s in dictionary." % (key,))

############################################################

import sys, time
def make_progress_logger(prefix):

    start_time = time.time()
    previous_time = start_time
    def f(p):
        current_time = time.time()
        # Don't write anything if you're calling this too fast.
        if (int(current_time - f.previous_time) > 2) and p > 0.0:

            finish_time = f.start_time + (current_time - f.start_time) / p
            time_left = finish_time - current_time

            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write("%s progress : %d %% . Estimated time left : %d secs." % (prefix, int(100*p), time_left))
            sys.stdout.flush()

            f.previous_time = current_time

    f.start_time = start_time
    f.previous_time = previous_time

    return f

############################################################

import numpy as np

def normalized_weighted_sum_with_log_coefficients(logc, E, axis=0):
    """
    E is a list of numpy arrays, or a numpy array itself.

    The point of this method is that we want to be
    able to sum E along one dimension with coefficients that
    are very small.

    If E is a list, then we don't have to specify the
    axis along which we are summing.
    """

    if type(E) == np.ndarray:
        assert len(logc.shape) == 1
        assert (logc.shape[0] == E.shape[axis])

        logc -= np.max(logc)

        A = np.array(E.shape)
        A[axis] = 1
        # let's say axis=3, then
        # A looks like (2,3,4,1,5,6)
        B = np.ones((len(E.shape),))
        B[axis] = logc.shape[0]
        # B looks like (1,1,1,n,1,1)
        c = np.exp(logc)
        C = np.tile(c.reshape(B), A)
        assert C.shape == E.shape
        res = (C * E).sum(axis=axis) / c.sum()
        return res
    elif type(E) == list:
        logc -= np.max(logc)
        c = np.exp(logc)
        return 1.0 / c.sum() * reduce(lambda x,y: x+y, E)
    else:
        raise("Error. Unrecognized format.")


def mvnpdf(x,m,covariance=None,precision=None,precision_det=None):
    return np.exp(log_mvnpdf(x,m,covariance,precision,precision_det=precision_det))

def log_mvnpdf(x,m,covariance=None,precision=None,precision_det=None):
    """
    Expecting either the covariance matrix or the precision matrix (its inverse).

    This function is not a vectorized function, nor does it perform
    some kind of caching to avoid performing the same matrix inversions repetitively.
    """
    assert (covariance is None) ^ (precision is None)
    assert len(x.shape) == 1
    assert len(m.shape) == 1
    d = x.shape[0]
    assert d == m.shape[0]

    if (precision is None):
        precision = np.linalg.inv(covariance)
        assert (d,d) == precision.shape

    if (precision_det is None):
        precision_det = np.linalg.det(precision)

    y = x - m
    return -0.5 * d * np.log(2*np.pi) + 0.5 * np.log(precision_det) - 0.5 * y.dot(precision).dot(y)


def grad_mvnpdf(x,m,covariance=None,precision=None,precision_det=None, want_log_decomposition=False):

    if (precision is None):
        precision = np.linalg.inv(covariance)

    if want_log_decomposition:
        # Returns two values instead which are intended to be used
        # with normalized_weighted_sum_with_log_coefficients(...).
        return (log_mvnpdf(x,m,precision=precision,precision_det=precision_det), -1.0 * precision.dot(x-m))
    else:
        return mvnpdf(x,m,precision=precision,precision_det=precision_det) * -1.0 * precision.dot(x-m)

####################################




