#!/usr/bin/env python
# encoding: utf-8
"""
dae.py

Guillaume Alain.
"""

import sys
import os
import pdb
import numpy

from theano import *
import theano.tensor as T

class DAE(object):
    """
    A DAE with tanh input units and tanh hidden units.
    """
    def __init__(self,
                 n_inputs=None,
                 n_hiddens=None,
                 W=None,
                 c=None,
                 b=None,
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
        W : array-like, shape (n_inputs, n_hiddens), optional
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
        if not (W == None):
            self.W = W
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
        # will have fields 'W', 'b', 'c', 'loss'


        # then setup the theano functions once
        self.theano_setup()
    
    def theano_setup(self):
    
        W = T.dmatrix('W')
        b = T.dvector('b')
        c = T.dvector('c')
        x = T.dmatrix('x')
    
        s = T.dot(x, W) + c
        # h = 1 / (1 + T.exp(-s))
        # h = T.nnet.sigmoid(s)
        h = T.tanh(s)
        # r = T.dot(h,W.T) + b
        # r = theano.printing.Print("r=")(2*T.tanh(T.dot(h,W.T) + b))
        ract = T.dot(h,W.T) + b
        r = self.output_scaling_factor * T.tanh(ract)
    
        #g  = function([W,b,c,x], h)
        #f  = function([W,b,c,h], r)
        #fg = function([W,b,c,x], r)
    
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

        self.theano_encode_decode = function([W,b,c,x], r)
        self.theano_loss = function([W,b,c,x,y], [loss, T.abs_(s), T.abs_(ract)])
        self.theano_gradients = function([W,b,c,x,y], [T.grad(sum_loss, W),
                                                       T.grad(sum_loss, b),
                                                       T.grad(sum_loss, c)])

    def encode_decode(self, x):
        return self.theano_encode_decode(self.W, self.b, self.c, x)

    def model_loss(self, x, noise_stddev = 0.0):
        """
        Computes the error of the model with respect
        to the total cost.
        
        -------
        x: array-like, shape (n_examples, n_inputs)
        
        Returns
        -------
        loss: array-like, shape (n_examples,)
        """

        if noise_stddev > 0.0:
            return self.theano_loss(self.W, self.b, self.c, x + numpy.random.normal(scale=noise_stddev, size=x.shape), x)
        else:
            return self.theano_loss(self.W, self.b, self.c, x, x)

    def reset_params(self):
        self.W = numpy.random.uniform( low = -0.1, high = 0.1, size=(self.n_inputs, self.n_hiddens) )
        #self.W = numpy.random.uniform(
        #    low = - 4.0 * numpy.sqrt(6./(self.n_inputs + self.n_hiddens)),
        #    high = 4.0 * numpy.sqrt(6./(self.n_inputs + self.n_hiddens)),
        #    size=(d, self.n_hiddens))
        self.c = numpy.zeros(self.n_hiddens)
        self.b = numpy.zeros(self.n_inputs)

    def set_params_to_best_noisy(self):
        self.W = self.best_noisy_params['W']
        self.b = self.best_noisy_params['b']
        self.c = self.best_noisy_params['c']

    def set_params_to_best_noiseless(self):
        self.W = self.best_noiseless_params['W']
        self.b = self.best_noiseless_params['b']
        self.c = self.best_noiseless_params['c']

    def perform_logging(self, X, noise_stddev, verbose = False):

        # The 'X' parameter is used to log the gradients.
        # We are recomputing them and wasting computation here, but
        # whenever we train a model we shouldn't be doing all the
        # extensive logging that we do for debugging purposes.
        # The 'X' is generally data from a minibatch.

        # Two blocks of code where the only word that changes is
        # 'noisy' to 'noiseless'.

        # 'noisy'
        noisy_all_losses, noisy_all_abs_act, noisy_all_abs_ract = self.model_loss(X, noise_stddev = noise_stddev)

        self.logging['noisy']['mean_abs_loss'].append( numpy.abs(noisy_all_losses).mean() )
        self.logging['noisy']['var_abs_loss'].append( numpy.abs(noisy_all_losses).var() )

        self.logging['noisy']['mean_abs_act'].append( noisy_all_abs_act.mean() )
        self.logging['noisy']['var_abs_act'].append( noisy_all_abs_act.var() )

        self.logging['noisy']['mean_abs_ract'].append( noisy_all_abs_ract.mean() )
        self.logging['noisy']['var_abs_ract'].append( noisy_all_abs_ract.var() )

        #if not (X == None):
        #    grad_W, grad_b, grad_c = self.one_step_grad_descent(X, perform_update = False)
        #    self.logging['noisy']['mean_abs_grad_W'].append( numpy.abs(grad_W).mean() )
        #    self.logging['noisy']['var_abs_grad_W'].append( numpy.abs(grad_W).var() )

        # if there is no key, or if we're beating the current best, replace the value
        if ((not self.best_noisy_params.has_key('loss')) or
            self.logging['noisy']['mean_abs_loss'][-1] < self.best_noisy_params['loss']):
                self.best_noisy_params['loss'] = self.logging['noisy']['mean_abs_loss'][-1]
                self.best_noisy_params['W'] = self.W
                self.best_noisy_params['b'] = self.b
                self.best_noisy_params['c'] = self.c

        # 'noiseless'
        noiseless_all_losses, noiseless_all_abs_act, noiseless_all_abs_ract = self.model_loss(X, noise_stddev = 0.0)

        self.logging['noiseless']['mean_abs_loss'].append( numpy.abs(noiseless_all_losses).mean() )
        self.logging['noiseless']['var_abs_loss'].append( numpy.abs(noiseless_all_losses).var() )

        self.logging['noiseless']['mean_abs_act'].append( noiseless_all_abs_act.mean() )
        self.logging['noiseless']['var_abs_act'].append(  noiseless_all_abs_act.var() )

        self.logging['noiseless']['mean_abs_ract'].append( noiseless_all_abs_ract.mean() )
        self.logging['noiseless']['var_abs_ract'].append( noiseless_all_abs_ract.var() )

        #if not (X == None):
        #    grad_W, grad_b, grad_c = self.one_step_grad_descent(X, perform_update = False, jacobi_penalty_override = 0.0)
        #    self.logging['noiseless']['mean_abs_grad_W'].append( numpy.abs(grad_W).mean() )
        #    self.logging['noiseless']['var_abs_grad_W'].append( numpy.abs(grad_W).var() )

        # if there is no key, or if we're beating the current best, replace the value
        if ((not self.best_noiseless_params.has_key('loss')) or
            self.logging['noiseless']['mean_abs_loss'][-1] < self.best_noiseless_params['loss']):
                self.best_noiseless_params['loss'] = self.logging['noiseless']['mean_abs_loss'][-1]
                self.best_noiseless_params['W'] = self.W
                self.best_noiseless_params['b'] = self.b
                self.best_noiseless_params['c'] = self.c


        if verbose:
            print "  -- Exact --"
            print "    Loss : %0.6f" % self.logging['noiseless']['mean_abs_loss'][-1]
            #print "    Activations Mean Abs. Hidden = %0.6f, Reconstructed = %0.6f" % (abs_act, abs_ract)
            print "  -- Noise --"
            print "    Loss : %0.6f" % self.logging['noisy']['mean_abs_loss'][-1]
            #print "    Activations Mean Abs. Hidden = %0.6f, Reconstructed = %0.6f" % (noise_abs_act, noise_abs_ract)
            #print "  Gradient W Mean Abs = %f" % numpy.abs(self.grad_W).mean()
            print "\n"



def main():
    pass


if __name__ == '__main__':
    main()
