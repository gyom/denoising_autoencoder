#!/usr/bin/env python
# encoding: utf-8
"""
dae.py

Guillaume Alain.
"""

import sys, os
import numpy as np

from dae import DAE

class DAE_interface_for_exact_grad_E(DAE):
    """
    This takes the gradient of an energy function and produces
    a structure that passes off as a DAE that satisfies some of
    the requirements for the sampling algorithms.

    It uses r(x) = x - grad(E) so you need to compensate with the
    proper scaling of langevin lambda.
    """
    def __init__(self, n_inputs, n_hiddens, grad_E, train_stddev)
        """
        Initialize a DAE.
        
        Parameters
        ----------
        n_inputs : int
            Number of inputs units
        n_hiddens : int
            Number of hidden units
        grad_E : function from R^n_inputs to R^n_inputs
            Not in a vectorial form.
        train_stddev : float
            Used to define the magnitude of the reconstruction
            function r with (r(x)-x)/train_stddev**2 = grad_E(x).
        """

        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens

        self.grad_E = grad_E
        self.train_stddev = train_stddev

        self.r = lambda x: x + train_stddev**2 * grad_E(x)

    #def encode(self, X):
    #    if X.shape[1] != self.n_inputs:
    #        raise("Using wrong shape[1] for the argument X to DAE.encode. It's %d when it should be %d" % (X.shape[1], self.n_inputs))
    #    return self.theano_encode(self.Wc, self.c, X)
    #
    #def decode(self, H):
    #    if H.shape[1] != self.n_hiddens:
    #        raise("Using wrong shape[1] for the argument H to DAE.decode. It's %d when it should be %d" % (H.shape[1], self.n_hiddens))
    #    return self.theano_decode(self.Wb, self.b, self.s, H)

    def encode_decode(self, X):
        if X.shape[1] != self.n_inputs:
            raise("Using wrong shape[1] for the argument X to DAE.encode_decode. It's %d when it should be %d" % (X.shape[1], self.n_inputs))
        
        return np.vstack([self.r(x) for x in X])

    def model_loss(self, X, noisy_X):
        """
        X:       array-like, shape (n_examples, n_inputs)
        noisy_X: array-like, shape (n_examples, n_inputs)

        Returns  loss: array-like, shape (n_examples,)
        """
        return np.vstack([np.linalg.norm(self.r(noisy_x) - x) for (x, noisy_x) in zip(X,noisy_X)])

    #def jacobian_encode(self, x):
    #    assert len(x.shape) == 1
    #    assert x.shape[0] == self.n_inputs
    #    return self.theano_jacobian_encode(self.Wc, self.c, x)

    def jacobian_encode_decode(self, x):
        assert len(x.shape) == 1
        assert x.shape[0] == self.n_inputs
        return np.eye(x.shape[0])

def main():
    pass


if __name__ == '__main__':
    main()
