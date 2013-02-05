

import numpy as np

# helper function
def cross_differences(true_samples, model_samples, f = lambda x: x**2):
    """
    true_samples of shape (N,d)
    model_samples of shape (M,d)

    Returns an array of shape (N,M).
    """

    assert true_samples.shape[1] == model_samples.shape[1]
    (N,d) = true_samples.shape
    (M,d) = model_samples.shape

    if N <= M:
        res = np.zeros((N,M))
        for n in np.arange(N):
            res[n,:] = f( np.tile(true_samples[n,:], (M,1)) - model_samples ).sum(axis=1)
        return res
    else:
        # switch the arguments and transpose
        # print "(N,M) = (%d,%d)" % (N,M)
        return cross_differences(model_samples, true_samples, f).T


# helper function
def pdf(true_samples, model_samples, stddev):
    """
    true_samples of shape (N,d)
    model_samples of shape (M,d)

    Returns an array of (N,).
    This function should be seen as a vectorial function.
    """
    assert true_samples.shape[1] == model_samples.shape[1]
    (N,d) = true_samples.shape
    (M,d) = model_samples.shape

    return 1 / np.sqrt(2*np.pi) / (stddev**d) * np.exp( - 0.5 * cross_differences(true_samples, model_samples) / stddev**2 ).mean(axis=1)


# helper function
def fit_stddev(true_samples, model_samples, initial_stddev):
    """
    Some kind of bisection search towards zero.
    Divide the stddev until you stop decreasing the likelihood.
    """

    current_stddev = initial_stddev
    current_loglik = np.log(pdf(true_samples, model_samples, current_stddev)).mean()

    # Python doesn't have a do-while control flow structure.
    # We have to fake one.
    proposed_stddev = current_stddev / 2.0
    proposed_loglik = np.log(pdf(true_samples, model_samples, proposed_stddev)).mean()

    while current_loglik < proposed_loglik:
        print "We just accepted the stddev %f transitioning from loglikelihood %f to %f." % (proposed_stddev, current_loglik, proposed_loglik)
        # accept the proposed stddev
        current_stddev = proposed_stddev
        current_loglik = proposed_loglik
        # propose a new one
        proposed_stddev = current_stddev / 2.0
        proposed_loglik = np.log(pdf(true_samples, model_samples, proposed_stddev)).mean()

    return current_stddev


# main function provided by this module
def KL(true_samples, model_samples, true_samples_Ntrain, true_samples_Ntest, stddev = None):

    assert type(true_samples) == np.ndarray
    assert type(model_samples) == np.ndarray

    assert true_samples_Ntrain + true_samples_Ntest <= true_samples.shape[0]
    ind = np.arange(0,true_samples.shape[0])
    np.random.shuffle(ind)

    if stddev == None:
        # train the stddev
        stddev = fit_stddev(true_samples[ind[0:true_samples_Ntrain],:], model_samples, 1.0)

    # Now compute the KL divergence approximation.
    # One might argue that the true_samples used to train
    # the stddev should not be used for the next step,
    # but we're using so many sample points that it should
    # be safe (given that we have only one parameter).

    KL_divergence = (   np.log(pdf(true_samples[ind[0:true_samples_Ntest],:], true_samples[ind[0:true_samples_Ntest],:], stddev)).mean()
                      - np.log(pdf(true_samples[ind[0:true_samples_Ntest],:], model_samples,                             stddev)).mean() )

    return KL_divergence



###
