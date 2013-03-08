
############################################################

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

def mvnpdf(x,m,covariance=None,precision=None):
    return np.exp(log_mvnpdf(x,m,covariance,precision))

def log_mvnpdf(x,m,covariance=None,precision=None):
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

    #print covariance.shape
    #print precision.shape
    assert (d,d) == precision.shape

    y = x - m
    return -0.5 * d * np.log(2*np.pi) + 0.5 * np.log(np.linalg.det(precision)) - 0.5 * y.dot(precision).dot(y)


def grad_mvnpdf(x,m,covariance=None,precision=None):

    if (precision is None):
        precision = np.linalg.inv(covariance)

    return mvnpdf(x,m,precision=precision) * -1.0 * precision.dot(x-m)

