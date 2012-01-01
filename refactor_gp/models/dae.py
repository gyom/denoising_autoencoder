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

import refactor_gp
import refactor_gp.gyom_utils
from   refactor_gp.gyom_utils import conj
from   refactor_gp.gyom_utils import make_progress_logger
from   refactor_gp.gyom_utils import isotropic_gaussian_noise_and_importance_sampling_weights


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

    def q_grad(self, q, X, noisy_X, importance_sampling_weights):
        raise("Abstract method")

    def q_loss(self, q, X, noisy_X, importance_sampling_weights):
        raise("Abstract method")


    def fit(self,
            X, noisy_X, importance_sampling_weights,
            optimization_args):
        """
        Fit the model to the data X.
        
        Parameters
        ----------
            X: array-like, shape (n_examples, n_inputs)
                Training data, where n_examples in the number of examples
                and n_inputs is the number of features.

            noisy_X: array-like, shape (n_examples, n_inputs)

            importance_sampling_weights : optional, array, shape (n_examples,)
        """

        if importance_sampling_weights is None:
            importance_sampling_weights = np.ones((X.shape[0],))

        def U(q):
            # because theano_loss is a vectorial function, we have to sum()
            return self.q_loss(q, X, noisy_X, importance_sampling_weights).sum()

        def grad_U(q):
            # because theano_gradients is NOT a vectorial function, no need so sum()
            return self.q_grad(q, X, noisy_X, importance_sampling_weights)

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

    def fit_with_stddevs_sequence(self, X, X_valid, stddevs,
                                  optimization_args):

        """
        stddevs has fields 'train', 'valid' and any number of other variants on 'valid'.
        The special key is 'train', used for training.
        The validation errors are computed with all the other keys that contain
        information about the stddev. Obviously, we want to use one called 'valid', but
        we can also have different alternatives such as 'alt_valid' or 'valid2' with
        a different sequence of stddevs.

        stddevs is of the form {'train' : [{'target' : 1.0, 'sampled' : 4.0},
                                           {'target' : 0.8, 'sampled' : 3.0},
                                           ...
                                           ],
                                'valid' : [{'target' : 1.0, 'sampled' : 4.0},
                                           {'target' : 0.8, 'sampled' : 3.0},
                                           ...
                                           ],
                                ...
                                }

        X is an array of shape (n_train, d)
        X_valid is an array of shape (n_valid, d). It can be None.

        optimisation_args passed through to the method 'fit' of this class.
        example of optimation_args :
                                     {'method' : 'fmin_l_bfgs_b',
                                      'maxiter' : maxiter,
                                      'm':lbfgs_rank}

        Returns the losses for all the stddevs. The variable 'best_q_mean_losses'.
        """

        def validate_the_stddevs_argument(stddevs):
            """
            Let's just validate the thing for now and add flexibility later.
            Note : This function gets everything from its arguments (not using closure).
            """
            assert stddevs.has_key('train')
            M = len(stddevs['train'])
            for key in stddevs.keys():
                assert type(stddevs[key]) == type([])
                assert M == len(stddevs[key])
                # turn any string "None" into a real None
                stddevs[key] = [None if e == "None" else e for e in stddevs[key]]
                for e in stddevs[key]:
                    # Allow for e to be None.
                    # For example, if you only want the validation errors
                    # for the final iteration, you don't want to be computing
                    # it at every step along the way. This is what None is for.
                    if e is not None:
                        assert e.has_key('target')
                        assert e.has_key('sampled')

        validate_the_stddevs_argument(stddevs)

        progress_logger = make_progress_logger("Training")


        best_q_mean_losses = dict([(key, []) for key in stddevs.keys()])
        # Summary :
        #     Everything that follows is just a way to mutate the value of 'best_q'.
        #     That 'best_q' variable contains the learned parameters.
        #     We log various things based on the current value of 'best_q' and
        #     the datasets (X, X_valid).
        #     At the end of the day, we're left with 'best_q' and stuff logged
        #     in 'best_q_mean_losses' to make an informed decision about the
        #     usefulness of the model learned.

        M = len(stddevs['train'])
        for m in range(0, M):

            train_target_stddev = stddevs['train'][m]['target']
            train_sampled_stddev = stddevs['train'][m]['sampled']

            sys.stdout.write("    Using train_stddev (target, sampled) = (%f, %f), " % (train_target_stddev, train_sampled_stddev)
            (noisy_X, importance_sampling_weights) = isotropic_gaussian_noise_and_importance_sampling_weights(X, train_target_stddev, train_sampled_stddev)

            (best_q, train_U_best_q) = self.fit(X, noisy_X, importance_sampling_weights, optimization_args)

            train_mean_U_best_q = train_U_best_q / X.shape[0]
            best_q_mean_losses['train'].append(train_mean_U_best_q)
            sys.stdout.write("train mean loss is %f, " % (train_mean_U_best_q,))

        
            if X_valid is not None:

                for key in stddevs.keys():
                    if key == 'train':
                        continue
                
                    some_valid_target_stddev = stddevs[key][m]['target']
                    some_valid_sampled_stddev = stddevs[key][m]['sampled']

                    if some_valid_sampled_stddev is None:
                        # it's fine if some_valid_target_stddev is None
                        # so we're not testing for that
                        best_q_mean_losses[key].append(None)
                        continue

                    sys.stdout.write("    Using %s stddev (target, sampled) = (%f, %f), " % (str(key), some_valid_target_stddev, some_valid_sampled_stddev)
                    (noisy_X_valid, importance_sampling_weights) = isotropic_gaussian_noise_and_importance_sampling_weights(X_valid, some_valid_sampled_stddev, some_valid_target_stddev)

                    some_valid_U_best_q = self.q_loss(best_q, X, noisy_X_valid, importance_sampling_weights).sum()

                    # Notice : Despite the importance_sampling_weights being used,
                    # I think that we are still doing the right thing by normalizing by
                    # X_valid.shape[0]. I was a bit afraid that we'd be throwing off everything
                    # by using these coefficients, but now I think that we won't find ourselves
                    # in a situation where the validation loss will be useless because of the
                    # wild importance sampling weights.

                    some_valid_mean_U_best_q = some_valid_U_best_q / X_valid.shape[0]
                    best_q_mean_losses[key].append(some_valid_mean_U_best_q)
                    sys.stdout.write("%s mean loss is %f, " % (str(key), some_valid_mean_U_best_q,))

                    progress_logger(1.0 * (m+1) / M)

        return best_q_mean_losses

    def fit_with_decreasing_noise(self, X, list_of_train_stddev,
                                  optimization_args, early_termination_args = {}, X_valid = None, list_of_additional_valid_stddev = None):
        """
        The 'optimization_args' filters through to the 'fit' function almost unchanged.

        There is the option of adding a a special provision
        for it's 'maxiter' entry when we get a list.
        In such a situation, we use one value of maxiter
        from the list for each value of list_of_train_stddev.

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
        progress_logger = make_progress_logger("Training")

        for train_stddev in list_of_train_stddev:

            sys.stdout.write("    Using train_stddev %f, " % train_stddev)
            (noisy_X, importance_sampling_weights) = isotropic_gaussian_noise_and_importance_sampling_weights(X, 4.0*train_stddev, train_stddev)
            #noisy_X = X + np.random.normal(size = X.shape, scale = train_stddev)

            if optimization_args.has_key('maxiter') and type(optimization_args['maxiter']) in [list, np.array]:
                assert len(optimization_args['maxiter']) == len(list_of_train_stddev)
                optimization_args0 = conj(optimization_args, "maxiter", optimization_args['maxiter'][i])
            else:
                optimization_args0 = optimization_args
            (best_q, train_U_best_q_) = self.fit(X, noisy_X, optimization_args0)
            #(best_q, train_U_best_q_) = self.fit(X, noisy_X, optimization_args)

            train_U_best_q = self.q_loss(best_q, X, noisy_X, importance_sampling_weights).sum()
            # sanity check to make sure that we're evaluating this right
            assert(abs(train_U_best_q - train_U_best_q_) < 1e-8)

            train_mean_U_best_q = train_U_best_q / X.shape[0]
            seq_train_mean_best_U_q.append(train_mean_U_best_q)
            sys.stdout.write("train mean loss is %f, " % (train_mean_U_best_q,))

            if not (X_valid == None):
                (noisy_X_valid, importance_sampling_weights) = isotropic_gaussian_noise_and_importance_sampling_weights(X_valid, 4.0*train_stddev, train_stddev)

                #noisy_X_valid = X_valid + np.random.normal(size = X_valid.shape, scale = train_stddev)
                valid_U_best_q = self.q_loss(best_q, X_valid, noisy_X_valid, importance_sampling_weights).sum()
                valid_mean_U_best_q = valid_U_best_q / X_valid.shape[0]
                seq_valid_mean_best_U_q.append(valid_mean_U_best_q)
                sys.stdout.write("valid mean loss is %f." % (valid_mean_U_best_q,))

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

            print ""
            progress_logger(1.0 * i / len(list_of_train_stddev))
            i += 1
        # end for

        # might as well pad the rest of the list to
        # signify that we terminated early
        while len(seq_train_mean_best_U_q) < len(list_of_train_stddev):
            seq_train_mean_best_U_q.append(np.nan)
        while len(seq_valid_mean_best_U_q) < len(list_of_train_stddev):
            seq_valid_mean_best_U_q.append(np.nan)


        # Now we want to recompute the model losses for all the values of
        # the train_stddev, but using the final parameters best_q.
        # This will be used as an addition quality evaluation to determine
        # how the DAE treats data that's relatively far from the manifold
        # once it's done training.
        # It might be even more informative than the validation losses.
        
        seq_valid_mean_U_final_best_q = None
        seq_alt_valid_mean_U_final_best_q = None
        if not (X_valid == None):
            nreps = 10
            # This thing doesn't work with the list comprehension. You need to generate the data every time.
            (noisy_X_valid, importance_sampling_weights) = isotropic_gaussian_noise_and_importance_sampling_weights(X_valid, 4.0*train_stddev, train_stddev)
            seq_valid_mean_U_final_best_q = [np.array([self.q_loss(best_q,
                                                                   X_valid,
                                                                   noisy_X_valid,
                                                                   importance_sampling_weights).sum() / X_valid.shape[0]
                                                       for _ in range(nreps)]).mean()
                                             for train_stddev in list_of_train_stddev]

            if (list_of_additional_valid_stddev is not None) and len(list_of_additional_valid_stddev) > 0:
                # TODO : use some kind of tool to generate the importance_sampling_weights
                seq_alt_valid_mean_U_final_best_q = [np.array([self.q_loss(best_q,
                                                                           X_valid,
                                                                           X_valid + np.random.normal(size = X_valid.shape, scale = alt_valid_stddev)).sum() / X_valid.shape[0]
                                                            for _ in range(nreps)]).mean()
                                                     for alt_valid_stddev in list_of_additional_valid_stddev]
        # end if

        return (seq_train_mean_best_U_q, seq_valid_mean_best_U_q, seq_valid_mean_U_final_best_q, seq_alt_valid_mean_U_final_best_q)

def main():
    pass


if __name__ == '__main__':
    main()
