
import numpy as np
import scipy.optimize
# for flushing output
import sys


# Helper functions. One being the opposite of the other.
def serialize_params_as_q(Wb, Wc, b, c):
    return np.hstack((Wb.reshape((-1,)),
                      Wc.reshape((-1,)),
                      b.reshape((-1,)),
                      c.reshape((-1,)))).reshape((-1,))

def read_params_from_q(q, n_inputs, n_hiddens):

    n_elems_Wb = n_inputs * n_hiddens
    n_elems_Wc = n_inputs * n_hiddens
    n_elems_b = n_inputs
    n_elems_c = n_hiddens

    bounds = (0,
              n_elems_Wb,
              n_elems_Wb + n_elems_Wc,
              n_elems_Wb + n_elems_Wc + n_elems_b,
              n_elems_Wb + n_elems_Wc + n_elems_b + n_elems_c)

    Wb = q[ bounds[0] : bounds[1] ].reshape((n_inputs, n_hiddens)).copy()
    Wc = q[ bounds[1] : bounds[2] ].reshape((n_inputs, n_hiddens)).copy()
    b  = q[ bounds[2] : bounds[3] ].reshape((n_elems_b,)).copy()
    c  = q[ bounds[3] : bounds[4] ].reshape((n_elems_c,)).copy()
        
    return (Wb, Wc, b, c)



def fit(the_dae,
        X, noisy_X,
        tolerance = 1.0e-6,
        verbose=False):
    """
    Fit the model to the data X.
    
    Parameters
    ----------
        X: array-like, shape (n_examples, n_inputs)
        Training data, where n_examples in the number of examples
        and n_inputs is the number of features.

        noisy_X: array-like, shape (n_examples, n_inputs)
    """

    # While this is only an implementation issue, note that this method
    # doesn't use the DAE at all during the training. It uses the methods
    # 'theano_loss' and 'theano_gradients', but both of those functions
    # are expected to be supplied all the required arguments.
    #
    # None of the arguments are read off as members of 'the_dae' except
    # at the beginning when getting the initial values and at the end
    # when we want the final values to be written back to the member variables.

    def f(q):
        # The dataset X is baked into the definition of U(q).
        (Wb, Wc, b, c) = read_params_from_q(q, the_dae.n_inputs, the_dae.n_hiddens)
        (loss, _, _) = the_dae.theano_loss(Wb, Wc, b, c, noisy_X, X)
        return loss.sum()

    def fprime(q):
        (Wb, Wc, b, c) = read_params_from_q(q, the_dae.n_inputs, the_dae.n_hiddens)
        (grad_Wb, grad_Wc, grad_b, grad_c) = the_dae.theano_gradients(Wb, Wc, b, c, noisy_X, X)
        # We can use the same function to spin the gradients into the
        # momentum vector. This works well because
        #   grad_Wb.shape == Wb.shape
        #   grad_Wc.shape == Wc.shape
        #   grad_b.shape == b.shape
        #   grad_c.shape == c.shape
        return serialize_params_as_q(grad_Wb, grad_Wc, grad_b, grad_c)

    # the initial state
    q0 = serialize_params_as_q(the_dae.Wb, the_dae.Wc, the_dae.b, the_dae.c)

    # Fuck everything about using global variables
    # like this because of python 2.7's broken lexical scoping rules.
    global callback_counter
    callback_counter = 0

    def loggin_callback(current_q):
        global callback_counter
        if the_dae.want_logging:
            # It's kinda funny that we need to write the parameters
            # into the DAE in order to perform the logging because we
            # make a call to its internal function that keeps track of the loss.
            (the_dae.Wb, the_dae.Wc, the_dae.b, the_dae.c) = read_params_from_q(current_q, the_dae.n_inputs, the_dae.n_hiddens)
            if verbose and (callback_counter % 100 == 0):
                sys.stdout.flush()
                print "Epoch %d of some scipy.optimize minimizer algorithm." % callback_counter
                the_dae.perform_logging(X, noisy_X = noisy_X, verbose = True)
                #the_dae.perform_logging(X, noise_stddev, verbose = True)
            else:
                the_dae.perform_logging(X, noisy_X = noisy_X, verbose = False)
                #the_dae.perform_logging(X, noise_stddev, verbose = False)
        callback_counter = callback_counter + 1

    # With everything set up, perform the optimization.
    best_q = scipy.optimize.fmin_cg(f, q0, fprime, callback = loggin_callback, gtol = tolerance)
    # Write back the best solution into the DAE.
    (the_dae.Wb, the_dae.Wc, the_dae.b, the_dae.c) = read_params_from_q(best_q, the_dae.n_inputs, the_dae.n_hiddens)
