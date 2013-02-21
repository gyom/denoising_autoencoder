#!/usr/bin/env python
# encoding: utf-8
"""
dae.py

Guillaume Alain.
"""

import sys
import os
import pdb

import numpy as np
import scipy

from theano import *
import theano.tensor as T

class DAE(object):

    #def __init__(self):
    #    pass

    #def encode(self, X):
    #    error("Abstract method")

    #def decode(self, H):
    #    error("Abstract method")

    #def encode_decode(self, X):
    #    error("Abstract method")

    #def model_loss(self, X, noisy_X):
    #    error("Abstract method")



    def q_read_params(self):
        raise("Abstract method")

    def q_set_params(self, q):
        raise("Abstract method")

    def q_grad(self, q, X, noisy_X):
        raise("Abstract method")

    def q_loss(self, q, X, noisy_X):
        raise("Abstract method")



    def fit(self,
            X, noisy_X,
            optimization_args):
        """
        Fit the model to the data X.
        
        Parameters
        ----------
            X: array-like, shape (n_examples, n_inputs)
                Training data, where n_examples in the number of examples
                and n_inputs is the number of features.

            noisy_X: array-like, shape (n_examples, n_inputs)
        """

        #print X.shape
        #print noisy_X.shape

        def U(q):
            # because theano_loss is a vectorial function, we have to sum()
            return self.q_loss(q, X, noisy_X).sum()

        def grad_U(q):
            # because theano_gradients is NOT a vectorial function, no need so sum()
            return self.q_grad(q, X, noisy_X)

        # Read the initial state from q.
        # This means that we can call the "fit" function
        # repeatedly with various (X, noisy_X) and we will
        # always be building on the current solution.
        q0 = self.q_read_params()

        global callback_counter
        callback_counter = 0
        def logging_callback(current_q):
            global callback_counter
            if callback_counter % 10 == 0:
                print "Iteration %d. Loss : %f" % (callback_counter, U(current_q))
            callback_counter = callback_counter + 1

        # With everything set up, perform the optimization.
        if optimization_args['method'] == 'fmin_cg':
            best_q = scipy.optimize.fmin_cg(U, q0, grad_U,
                                            callback = logging_callback,
                                            gtol = optimization_args['gtol'],
                                            maxiter = optimization_args['maxiter'])
        elif optimization_args['method'] == 'fmin_ncg':
            best_q = scipy.optimize.fmin_ncg(U, q0, grad_U,
                                             callback = logging_callback,
                                             avextol = optimization_args['avextol'],
                                             maxiter = optimization_args['maxiter'])
        elif optimization_args['method'] == 'fmin_bfgs':
            best_q = scipy.optimize.fmin_bfgs(U, q0, grad_U,
                                              callback = logging_callback,
                                              maxiter = optimization_args['maxiter'])
        elif optimization_args['method'] == 'fmin_l_bfgs_b':
            # Cannot perform the logging.
            (best_q, _, details) = scipy.optimize.fmin_l_bfgs_b(U, q0, grad_U,
                                                                m = optimization_args['m'],
                                                                maxfun = optimization_args['maxiter'])
        else:
            assert False, "Unrecognized method name : " + optimization_args['method']

        # Don't forget to set the params after you optimized them !
        assert type(best_q) == np.ndarray
        self.q_set_params(best_q)
        return (best_q, U(best_q))

    def fit_with_decreasing_noise(self, X, list_of_train_stddev,
                                  optimization_args):
        seq_mean_best_U_q = []
        for train_stddev in list_of_train_stddev:
            sys.stdout.write("Using train_stddev %f, " % train_stddev)
            noisy_X = X + np.random.normal(size = X.shape, scale = train_stddev)
            (_, U_best_q) = self.fit(X, noisy_X, optimization_args)
            sys.stdout.write("mean loss is %f" % (U_best_q / X.shape[0]))
            seq_mean_best_U_q.append(U_best_q / X.shape[0])
            print ""
            # Should we log the optimization results in some way here
            # so as to print graphs showing the state of the learned
            # density as we decrease train_stddev ?

        return seq_mean_best_U_q


def main():
    pass


if __name__ == '__main__':
    main()
