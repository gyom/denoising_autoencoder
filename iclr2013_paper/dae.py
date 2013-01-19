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

from theano import *
import theano.tensor as T

class DAE(object):
    """
    A DAE with tanh input units and tanh hidden units.
    """
    def __init__(self,
                 n_inputs=None,
                 n_hiddens=None,
                 Wc=None, Wb=None,
                 c=None, b=None,
                 output_scaling_factor=1.0,
                 want_logging=True):
        """
        Initialize a DAE.
        
        Parameters
        ----------
        n_inputs : int
            Number of inputs units
        n_hiddens : int
            Number of hidden units
        Wc : array-like, shape (n_inputs, n_hiddens), optional
             Weight matrix, where n_inputs in the number of input
             units and n_hiddens is the number of hidden units.
        Wb : array-like, shape (n_inputs, n_hiddens), optional
             Weight matrix, where n_inputs in the number of input
             units and n_hiddens is the number of hidden units.
        c : array-like, shape (n_hiddens,), optional
            Biases of the hidden units
        b : array-like, shape (n_inputs,), optional
            Biases of the input units
        output_scaling_factor : real
            Applied after the second tanh at the output.
            Allows us to represent values in a range [-4,4]
            instead of just [-1,1] by using output_scaling_factor = 4.0.
        """

        # These values are to be treated as READ-ONLY.
        self.output_scaling_factor = output_scaling_factor
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens

        # These values are expected to be modified by
        # algorithms that take the DAE instance as parameter.
        # ex : any training function
        self.reset_params()
        if not (Wc == None):
            self.Wc = Wc
        if not (Wc == None):
            self.Wb = Wb
        if not (c == None):
            self.c = c
        if not (b == None):
            self.b = b

        self.want_logging = want_logging

        # logging
        
        if self.want_logging:
            self.logging = {}
            for k in ['noisy', 'noiseless']:
                self.logging[k] = {}
                self.logging[k]['mean_abs_loss'] = []
                self.logging[k]['var_abs_loss'] = []

                self.logging[k]['mean_abs_act'] = []
                self.logging[k]['var_abs_act'] = []

                self.logging[k]['mean_abs_ract'] = []
                self.logging[k]['var_abs_ract'] = []

                self.logging[k]['mean_abs_grad_W'] = []
                self.logging[k]['var_abs_grad_W'] = []

        # keep the best parameters
        self.best_noisy_params = {}
        self.best_noiseless_params = {}
        # will have fields 'Wb', 'Wc', 'b', 'c', 'loss'


        # then setup the theano functions once
        self.theano_setup()
    
    def theano_setup(self):
    
        # The matrices Wb and Wc were originally tied.
        # Because of that, I decided to keep Wb and Wc with
        # the same shape (instead of being transposed) to
        # avoid disturbing the code as much as possible.

        Wb = T.dmatrix('Wb')
        Wc = T.dmatrix('Wc')
        b = T.dvector('b')
        c = T.dvector('c')
        x = T.dmatrix('x')
    
        s = T.dot(x, Wc) + c
        # h = 1 / (1 + T.exp(-s))
        # h = T.nnet.sigmoid(s)
        h = T.tanh(s)
        # r = T.dot(h,W.T) + b
        # r = theano.printing.Print("r=")(2*T.tanh(T.dot(h,W.T) + b))
        ract = T.dot(h, Wb.T) + b
        r = self.output_scaling_factor * T.tanh(ract)
    
        # Another variable to be able to call a function
        # with a noisy x and compare it to a reference x.
        y = T.dmatrix('y')

        loss = ((r - y)**2)
        sum_loss = T.sum(loss)
        
        # theano_encode_decode : vectorial function in argument X.
        # theano_loss : vectorial function in argument X.
        # theano_gradients : returns triplet of gradients, each of
        #                    which involves the all data X summed
        #                    so it's not a "vectorial" function.

        self.theano_encode_decode = function([Wb,Wc,b,c,x], r)
        self.theano_loss = function([Wb,Wc,b,c,x,y], [loss, T.abs_(s), T.abs_(ract)])
        self.theano_gradients = function([Wb,Wc,b,c,x,y],
                                         [T.grad(sum_loss, Wb), T.grad(sum_loss, Wc),
                                          T.grad(sum_loss, b),  T.grad(sum_loss, c)])

        # other useful theano functions for the experiments that involve
        # adding noise to the hidden states
        self.theano_encode = function([Wc,c,x], h)
        self.theano_decode = function([Wb,b,h], r)

        # A non-vectorial implementation of the jacobian
        # of the encoder. Meant to be used with only one x
        # at a time, returning a matrix.
        jacob_x = T.dvector('jacob_x')
        jacob_c = T.dvector('jacob_c')
        jacob_Wc = T.dmatrix('jacob_Wc')
        jacob_s = T.dot(jacob_x, jacob_Wc) + jacob_c
        jacob_h = T.tanh(jacob_s)
        self.theano_encoder_jacobian_single = function([jacob_Wc,jacob_c,jacob_x], gradient.jacobian(jacob_h,jacob_x,consider_constant=[jacob_Wc,jacob_c]))



    def encode(self, X):
        if X.shape[1] != self.n_inputs:
            error("Using wrong shape[1] for the argument X to DAE.encode. It's %d when it should be %d" % (X.shape[1], self.n_inputs))
        return self.theano_encode(self.Wc, self.c, X)

    def decode(self, H):
        if H.shape[1] != self.n_hiddens:
            error("Using wrong shape[1] for the argument H to DAE.decode. It's %d when it should be %d" % (H.shape[1], self.n_hiddens))
        return self.theano_decode(self.Wb, self.b, H)

    def encode_decode(self, X):
        if X.shape[1] != self.n_inputs:
            error("Using wrong shape[1] for the argument X to DAE.encode_decode. It's %d when it should be %d" % (X.shape[1], self.n_inputs))
        return self.theano_encode_decode(self.Wb, self.Wc, self.b, self.c, X)

    def encoder_jacobian(self, X):

        error("This function appears to be broken. I don't know why, but just don't use it.")

        # We are exploiting the fact that, for a single x we have
        #
        # $\frac{\partial h}{\partial x}=\textrm{diag}\left(1-tanh^{2}(xW_{c}+c)\right)W_{c}^{T}$
        #
        # which means that we have to compute
        #   the hidden units H of shape (N, n_hiddens), reshaped as (N, n_hiddens, 1)
        #   the matrix Wc of shape (n_hiddens, n_inputs), reshape as (1, n_hiddens, n_inputs)
        #
        # which we then combine by broadcasting along the new dimensions.

        #N = X.shape[0]
        #
        #H = self.encode(X).reshape((N,self.n_hiddens,1))
        #WT = self.Wc.T.reshape((1,self.n_hiddens,self.n_inputs))
        #
        #return np.tile(1- H**2, (1,1,self.n_inputs)) * np.tile(WT, (N,1,1))

    def encoder_jacobian_single(self, x):

        if len(x.shape) == 2 and x.shape[1] == self.n_inputs:
            return self.theano_encoder_jacobian_single(self.Wc, self.c, x.reshape((-1,)))
        elif len(x.shape) == 1 and x.shape[0] == self.n_inputs:
            return self.theano_encoder_jacobian_single(self.Wc, self.c, x)
        else:
            error("You are misusing the sanity check function encoder_jacobian_sanity by giving an argument x of wrong shape.")


    def model_loss(self, X, noisy_X = None, noise_stddev = 0.0):
        """
        Computes the error of the model with respect
        to the total cost.
        
        -------
        x: array-like, shape (n_examples, n_inputs)
        
        Returns
        -------
        loss: array-like, shape (n_examples,)
        """

        # The preferred way is to specify noisy_X,
        # but we'll still go with the old noise_stddev.

        if noisy_X == None:
            if noise_stddev > 0.0:
                noisy_X = X + np.random.normal(scale = noise_stddev, size = X.shape)
            else:
                noisy_X = X
        
        return self.theano_loss(self.Wb, self.Wc, self.b, self.c, noisy_X, X)


    def reset_params(self):
        self.Wb = np.random.uniform( low = -1.0, high = 1.0, size=(self.n_inputs, self.n_hiddens) )
        self.Wc = np.random.uniform( low = -1.0, high = 1.0, size=(self.n_inputs, self.n_hiddens) )
        #self.W = np.random.uniform(
        #    low = - 4.0 * np.sqrt(6./(self.n_inputs + self.n_hiddens)),
        #    high = 4.0 * np.sqrt(6./(self.n_inputs + self.n_hiddens)),
        #    size=(d, self.n_hiddens))
        self.b = np.random.uniform( low = -0.1, high = 0.1, size=(self.n_inputs,) )
        self.c = np.random.uniform( low = -0.1, high = 0.1, size=(self.n_hiddens,) )
        #self.b = np.zeros(self.n_inputs)
        #self.c = np.zeros(self.n_hiddens)

    def set_params_to_best_noisy(self):
        self.Wb = self.best_noisy_params['Wb']
        self.Wc = self.best_noisy_params['Wc']
        self.b = self.best_noisy_params['b']
        self.c = self.best_noisy_params['c']

    def set_params_to_best_noiseless(self):
        self.Wb = self.best_noiseless_params['Wb']
        self.Wc = self.best_noiseless_params['Wc']
        self.b = self.best_noiseless_params['b']
        self.c = self.best_noiseless_params['c']

    def perform_logging(self, X, noisy_X, noise_stddev = None, verbose = False):

        # The 'X' parameter is used to log the gradients.
        # We are recomputing them and wasting computation here, but
        # whenever we train a model we shouldn't be doing all the
        # extensive logging that we do for debugging purposes.
        # The 'X' is generally data from a minibatch.

        # Two blocks of code where the only word that changes is
        # 'noisy' to 'noiseless'.

        # 'noisy'
        if noise_stddev == None:
            noisy_all_losses, noisy_all_abs_act, noisy_all_abs_ract = self.model_loss(X, noisy_X = noisy_X)
        else:
            noisy_all_losses, noisy_all_abs_act, noisy_all_abs_ract = self.model_loss(X, noise_stddev = noise_stddev)

        self.logging['noisy']['mean_abs_loss'].append( np.abs(noisy_all_losses).mean() )
        self.logging['noisy']['var_abs_loss'].append( np.abs(noisy_all_losses).var() )

        self.logging['noisy']['mean_abs_act'].append( noisy_all_abs_act.mean() )
        self.logging['noisy']['var_abs_act'].append( noisy_all_abs_act.var() )

        self.logging['noisy']['mean_abs_ract'].append( noisy_all_abs_ract.mean() )
        self.logging['noisy']['var_abs_ract'].append( noisy_all_abs_ract.var() )

        #if not (X == None):
        #    grad_W, grad_b, grad_c = self.one_step_grad_descent(X, perform_update = False)
        #    self.logging['noisy']['mean_abs_grad_W'].append( np.abs(grad_W).mean() )
        #    self.logging['noisy']['var_abs_grad_W'].append( np.abs(grad_W).var() )

        # if there is no key, or if we're beating the current best, replace the value
        if ((not self.best_noisy_params.has_key('loss')) or
            self.logging['noisy']['mean_abs_loss'][-1] < self.best_noisy_params['loss']):
                self.best_noisy_params['loss'] = self.logging['noisy']['mean_abs_loss'][-1]
                self.best_noisy_params['Wb'] = self.Wb
                self.best_noisy_params['Wc'] = self.Wc
                self.best_noisy_params['b'] = self.b
                self.best_noisy_params['c'] = self.c
                #print "Updated the best noisy loss as %0.6f" % self.logging['noisy']['mean_abs_loss'][-1]

        # 'noiseless'
        noiseless_all_losses, noiseless_all_abs_act, noiseless_all_abs_ract = self.model_loss(X, noisy_X = X)

        self.logging['noiseless']['mean_abs_loss'].append( np.abs(noiseless_all_losses).mean() )
        self.logging['noiseless']['var_abs_loss'].append( np.abs(noiseless_all_losses).var() )

        self.logging['noiseless']['mean_abs_act'].append( noiseless_all_abs_act.mean() )
        self.logging['noiseless']['var_abs_act'].append(  noiseless_all_abs_act.var() )

        self.logging['noiseless']['mean_abs_ract'].append( noiseless_all_abs_ract.mean() )
        self.logging['noiseless']['var_abs_ract'].append( noiseless_all_abs_ract.var() )

        #if not (X == None):
        #    grad_W, grad_b, grad_c = self.one_step_grad_descent(X, perform_update = False, jacobi_penalty_override = 0.0)
        #    self.logging['noiseless']['mean_abs_grad_W'].append( np.abs(grad_W).mean() )
        #    self.logging['noiseless']['var_abs_grad_W'].append( np.abs(grad_W).var() )

        # if there is no key, or if we're beating the current best, replace the value
        if ((not self.best_noiseless_params.has_key('loss')) or
            self.logging['noiseless']['mean_abs_loss'][-1] < self.best_noiseless_params['loss']):
                self.best_noiseless_params['loss'] = self.logging['noiseless']['mean_abs_loss'][-1]
                self.best_noiseless_params['Wb'] = self.Wb
                self.best_noiseless_params['Wc'] = self.Wc
                self.best_noiseless_params['b'] = self.b
                self.best_noiseless_params['c'] = self.c
                #print "Updated the best noiseless loss as %0.6f" % self.logging['noiseless']['mean_abs_loss'][-1]


        if verbose:
            #print "  -- Exact --"
            #print "    Loss : %0.6f" % self.logging['noiseless']['mean_abs_loss'][-1]
            #print "    Activations Mean Abs. Hidden = %0.6f, Reconstructed = %0.6f" % (abs_act, abs_ract)
            #print "  -- Noise --"
            print "    Loss : %0.10f" % self.logging['noisy']['mean_abs_loss'][-1]
            #print "    Activations Mean Abs. Hidden = %0.6f, Reconstructed = %0.6f" % (noise_abs_act, noise_abs_ract)
            #print "  Gradient W Mean Abs = %f" % np.abs(self.grad_W).mean()
            #print "\n"



def main():
    pass


if __name__ == '__main__':
    main()
