
# It's not even required to load the implementation
# of the DAE class. We're not doing object mocking
# at this point, but if we were that would mean that
# we wouldn't be able to load the DAE class implementation
# anyways.
#
# import dae

import numpy as np
# for the flush
import sys

def perform_one_update(the_dae, X, noisy_X, learning_rate, simulate_only  = False):
    """
    Perform one step of gradient descent on the
    DAE objective using the examples {\bf X}.
        
    Parameters
    ----------
        X: array-like, shape (n_examples, n_inputs)
    """
        
    #if noise_stddev > 0.0:
    #    perturbed_X = X + np.random.normal(scale = noise_stddev, size=X.shape)
    #else:
    #    perturbed_X = X.copy()

    grad_W, grad_b, grad_c = the_dae.theano_gradients(the_dae.W,
                                                      the_dae.b,
                                                      the_dae.c,
                                                      noisy_X,
                                                      X)

    if not simulate_only:
        the_dae.W = the_dae.W - learning_rate * grad_W
        the_dae.b = the_dae.b - learning_rate * grad_b
        the_dae.c = the_dae.c - learning_rate * grad_c

    return (grad_W, grad_b, grad_c)


def fit(the_dae, X, noisy_X, batch_size,
        n_epochs, learning_rate, verbose=False):
    """
    Fit the model to the data X.
    
    Parameters
    ----------
        X: array-like, shape (n_examples, n_inputs)
        Training data, where n_examples in the number of examples
        and n_inputs is the number of features.
    """

    # We don't necessarily want to reset the parameters before fitting.
    # the_dae.reset_params()

    # We'll be using double indirection to shuffle
    # around the minibatches. We will keep shuffling
    # the indices in 'inds' and the chunks will be
    # described by 'inds_ranges' which will be a collection
    # of ranges.
    #
    # ex :
    #    self.batch_size is 3
    #    inds is [10,3,2,5,9,8,6,0,7,4,1]
    #    inds_ranges is [(0,3), (3,6), (6,9), (9,10)]
    #
    # Results in batches being
    #    [10,3,2], [5,9,8], [0,7,4], [1]
    
    inds = range(X.shape[0])
        
    n_batches = len(inds) / batch_size
    inds_ranges = []
    for k in range(0, n_batches):

        start = k * batch_size
        if start >= X.shape[0]:
            break

        end = (k+1) * batch_size
        end = min(end, X.shape[0])

        # Keep in mind that the lower bound is inclusive
        # and the upper bound is exclusive.
        # This is why 'end' terminates with X.shape[0]
        # while that value would be illegal for 'start'.

        inds_ranges.append( (start, end) )
            
    if verbose:
        print "The ranges used for the minibatches are "
        print inds_ranges

    for epoch in range(n_epochs):

        # Shuffle the 'inds', because we don't modify
        # the 'inds_ranges'. Only one of them has to change.
        np.random.shuffle(inds)
            
        for (start, end) in inds_ranges:
            X_minibatch = X[inds[start:end]]
            noisy_X_minibatch = noisy_X[inds[start:end]]
            # calling the function defined here in
            # dae_train_gradient_descent.perform_one_update
            perform_one_update(the_dae, X_minibatch, noisy_X_minibatch, learning_rate)

        if the_dae.want_logging:
            if verbose and (epoch % 100 == 0):
                sys.stdout.flush()
                print "Epoch %d" % epoch
                the_dae.perform_logging(X, noisy_X = noisy_X, verbose = True)
                #the_dae.perform_logging(X, noise_stddev, verbose = True)
            else:
                the_dae.perform_logging(X, noisy_X = noisy_X, verbose = False)
                #the_dae.perform_logging(X, noise_stddev, verbose = False)
