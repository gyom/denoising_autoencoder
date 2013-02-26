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
                                  optimization_args, early_termination_args = {}, X_valid = None):
        """
        The 'optimization_args' filters through to the 'fit' function unchanged

        The 'early_termination_args' is optional. It provides a way to
        stop the training if we determine that we started in a state
        that was irredeemable and would only lead to a bad local minimum.
        We can keep in mind the r(x) = x solution as a benchmark and
        observe that, with r(x) = x we would have a loss function that
        roughly equals
            d * train_stddev**2, where d is the dimension of the data.

        The 'early_termination_args' dict has one key for now.
            early_termination_args['stop_if_loss_greater_than'] = [...]
                or
            early_termination_args['stop_if_loss_greater_than'] = "auto"

        If X_valid is not None, we will also return the values of the
        objective function evaluated with those validation samples.
        Those values will be the onces according to which we will
        decide to stop or not the descent with the train_stddev values.
        """

        # If we were passed the argument "auto", we have to replace the
        # value with an array of corresponding values.
        if (early_termination_args.has_key('stop_if_loss_greater_than') and type(early_termination_args['stop_if_loss_greater_than']) == str):
            if early_termination_args['stop_if_loss_greater_than'] == "auto":
                early_termination_args['stop_if_loss_greater_than'] = [X.shape[1] * train_stddev**2 for train_stddev in list_of_train_stddev]
                print "early termination with losses : "
                print early_termination_args['stop_if_loss_greater_than']
            else:
                print "Wrong value for early_termination_args. Only valid string is 'auto'."
                print "Exiting."
                quit()

        # at some point we might want to decide to
        # record all the best_q for the sequence
        seq_train_mean_best_U_q = []
        seq_valid_mean_best_U_q = []
        i = 0
        for train_stddev in list_of_train_stddev:

            sys.stdout.write("Using train_stddev %f, " % train_stddev)
            noisy_X = X + np.random.normal(size = X.shape, scale = train_stddev)
            (best_q, train_U_best_q_) = self.fit(X, noisy_X, optimization_args)

            train_U_best_q = self.q_loss(best_q, X, noisy_X).sum()
            # sanity check to make sure that we're evaluating this right
            assert(abs(train_U_best_q - train_U_best_q_) < 1e-8)

            train_mean_U_best_q = train_U_best_q / X.shape[0]
            seq_train_mean_best_U_q.append(train_mean_U_best_q)
            sys.stdout.write("train mean loss is %f" % (train_mean_U_best_q,))
            print ""

            if not (X_valid == None):
                noisy_X_valid = X_valid + np.random.normal(size = X_valid.shape, scale = train_stddev)
                valid_U_best_q = self.q_loss(best_q, X_valid, noisy_X_valid).sum()
                valid_mean_U_best_q = valid_U_best_q / X_valid.shape[0]
                seq_valid_mean_best_U_q.append(valid_mean_U_best_q)
                sys.stdout.write("valid mean loss is %f" % (valid_mean_U_best_q,))
                print ""

                # if we're dealing with a validation set, it will be the one used
                # to determine the stopping point
                if (early_termination_args.has_key('stop_if_loss_greater_than') and
                early_termination_args['stop_if_loss_greater_than'][i] < valid_mean_U_best_q):
                    break
            else:
                # if we don't have a validation set, then we'll use mean_U_best_q
                # for the termination condition

                if (early_termination_args.has_key('stop_if_loss_greater_than') and
                    early_termination_args['stop_if_loss_greater_than'][i] < mean_U_best_q):
                    break

            i += 1
        # end for

        # might as well pad the rest of the list to
        # signify that we terminated early
        while len(seq_train_mean_best_U_q) < len(list_of_train_stddev):
            seq_train_mean_best_U_q.append(np.nan)
        while len(seq_valid_mean_best_U_q) < len(list_of_train_stddev):
            seq_valid_mean_best_U_q.append(np.nan)
        
        return (seq_train_mean_best_U_q, seq_valid_mean_best_U_q)

def main():
    pass


if __name__ == '__main__':
    main()
