

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
def pdf_alternative(true_samples, model_samples, stddev):
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

    # Double-vectoring doesn't buy much performance.
    # Might as well to through the list iterating.
    results = []
    for n in np.arange(N):
        results.append( 1 / np.sqrt(2*np.pi) / (stddev**d) * np.exp( - 0.5 * ((true_samples[n,:] - model_samples)**2).sum(axis=1) / stddev**2 ).mean() )
        # This was 
        #    results.append( 1 / np.sqrt(2*np.pi) / (stddev**d) * np.exp( - 0.5 * ((true_samples[n,:] - model_samples)**2).sum(axis=1) / stddev**2 ) )
        # before, but the returned results were (N,M) which was
        # too much memory consumption.
    return np.vstack(results)


def cross_entropy(true_samples, model_samples, stddev):
    # Was
    #    return - np.log( pdf(true_samples, model_samples, stddev).mean(axis=1) ).mean()
    # with the old implementation of pdf that returned an nd.array of shape (N,M).
    return - np.log( pdf(true_samples, model_samples, stddev) ).mean()


# helper function
def fit_stddev_halving(true_samples, model_samples, initial_stddev):
    """
    Some kind of bisection search towards zero.
    Divide the stddev until you stop decreasing the likelihood.
    """
    current_stddev = initial_stddev
    current_loglik = - cross_entropy(true_samples, model_samples, current_stddev)

    # Python doesn't have a do-while control flow structure.
    # We have to fake one.
    proposed_stddev = current_stddev / 2.0
    proposed_loglik = - cross_entropy(true_samples, model_samples, proposed_stddev)

    while current_loglik < proposed_loglik:
        print "We just accepted the stddev %f transitioning from loglikelihood %f to %f." % (proposed_stddev, current_loglik, proposed_loglik)
        # accept the proposed stddev
        current_stddev = proposed_stddev
        current_loglik = proposed_loglik
        # propose a new one
        proposed_stddev = current_stddev / 2.0
        proposed_loglik = - cross_entropy(true_samples, model_samples, proposed_stddev)

    return current_stddev



def fit_stddev_enumeration(true_samples, model_samples, stddev_array, verbose=False):
    """
    Some kind of bisection search towards zero.
    Divide the stddev until you stop decreasing the likelihood.
    """

    assert type(stddev_array) in [np.ndarray, type([])]

    N = len(stddev_array)
    logliks = np.ones((N,))* (-np.inf)
    for n in np.arange(N):
        logliks[n] = - cross_entropy(true_samples, model_samples, stddev_array[n])
        if verbose:
            print "stddev %f gives loglik %f" % (stddev_array[n], logliks[n])
        if logliks[n] == -np.inf:
            break

    return stddev_array[np.argmax(logliks)]



# main function provided by this module
def KL(true_samples, model_samples, true_samples_Ntrain = None, true_samples_Ntest = None, true_stddev = None, model_stddev = None, verbose = False):

    assert type(true_samples) == np.ndarray
    assert type(model_samples) == np.ndarray

    # we can accept proportions between [0.0, 1.0] as arguments
    if true_samples_Ntrain > 0.0 and true_samples_Ntrain < 1.0:
        true_samples_Ntrain = int(true_samples.shape[0] * true_samples_Ntrain)
    if true_samples_Ntest > 0.0 and true_samples_Ntest < 1.0:
        true_samples_Ntest = int(true_samples.shape[0] * true_samples_Ntest)

    # If one of the quantities is missing, find the appropriate value.
    # If both are missing, use half for each.
    if true_samples_Ntrain == None and true_samples_Ntest == None:
        true_samples_Ntrain == int(true_samples.shape[0]/2)
        true_samples_Ntest == true_samples.shape[0] - true_samples_Ntrain
    elif true_samples_Ntrain == None:
        true_samples_Ntrain == true_samples.shape[0] - true_samples_Ntest
    elif true_samples_Ntest == None:
        true_samples_Ntest == true_samples.shape[0] - true_samples_Ntrain


    assert true_samples_Ntrain + true_samples_Ntest <= true_samples.shape[0]
    ind = np.arange(0,true_samples.shape[0])
    np.random.shuffle(ind)

    if true_stddev == None:
        true_stddev = fit_stddev_enumeration(true_samples[ind[0:int(true_samples_Ntrain/2)],:],
                                             true_samples[ind[int(true_samples_Ntrain/2):true_samples_Ntrain],:],
                                             np.exp(np.linspace(0.0, -8.0, 20)),
                                             verbose)
        print "Using true_stddev = %f as found by optimization" % true_stddev
    else:
        print "Using true_stddev = %f given by parameter." % true_stddev

    if model_stddev == None:
        model_stddev = fit_stddev_enumeration(true_samples[ind[0:true_samples_Ntrain],:],
                                            model_samples,
                                            np.exp(np.linspace(0.0, -8.0, 20)),
                                            verbose)
        print "Using model_stddev = %f as found by optimization" % model_stddev
    else:
        print "Using model_stddev = %f given by parameter." % model_stddev


    # Now compute the KL divergence approximation.
    # One might argue that the true_samples used to train
    # the stddev should not be used for the next step,
    # but we're using so many sample points that it should
    # be safe (given that we have only one parameter).

    #KL_divergence = (   np.log(pdf(true_samples[ind[0:true_samples_Ntest],:], true_samples[ind[0:true_samples_Ntest],:], stddev)).mean()
    #                  - np.log(pdf(true_samples[ind[0:true_samples_Ntest],:], model_samples,                             stddev)).mean() )

    KL_divergence = ( - cross_entropy( true_samples[ind[true_samples_Ntrain:(true_samples_Ntrain+true_samples_Ntest)],:], true_samples[ind[true_samples_Ntrain:(true_samples_Ntrain+true_samples_Ntest)],:], true_stddev)
                      + cross_entropy( true_samples[ind[true_samples_Ntrain:(true_samples_Ntrain+true_samples_Ntest)],:], model_samples,                             model_stddev) )


    return KL_divergence

### end of this module ###


### a basic main function to be used ###

def usage():
    print "-- usage example --"


def main():
    """
    This main function is not expected to be used often,
    but it's pretty much a wrapper around the "KL" function
    above.

    It takes two arguments, which represent file names containing
    samples from the true distribution and from the model
    distribution.

    The notation used will be that we are interested in
    computing KL(p||q) = \int p log (p/q)
    """

    import os, sys, getopt
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["help", "p_samples_pkl=", "q_samples_pkl=", "p_stddev=", "q_stddev=", "p_Ntrain=", "p_Ntest="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    p_samples_pkl = None
    q_samples_pkl = None
    p_Ntrain = 0.5
    p_Ntest = 0.5
    p_stddev = None
    q_stddev = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            # unused
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--p_samples_pkl"):
            p_samples_pkl = a
            assert os.path.exists(p_samples_pkl)
        elif o in ("--q_samples_pkl"):
            q_samples_pkl = a
            assert os.path.exists(q_samples_pkl)
        elif o in ("--p_Ntrain"):
            p_Ntrain = int(a)
        elif o in ("--p_Ntest"):
            p_Ntest = int(a)
        elif o in ("--p_stddev"):
            p_stddev = float(a)
        elif o in ("--q_stddev"):
            q_stddev = float(a)
        else:
            assert False, "unhandled option"

    assert p_samples_pkl
    assert q_samples_pkl

    import cPickle
    p_samples = cPickle.load(open(p_samples_pkl))
    q_samples = cPickle.load(open(q_samples_pkl))
    
    # listing more stddev
    KL_value = KL(p_samples, q_samples, p_Ntrain, p_Ntest, p_stddev, q_stddev, verbose)
    print KL_value


if __name__ == "__main__":
    main()
