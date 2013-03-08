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
    def __init__(self, n_inputs, n_hiddens, grad_E)
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
        """

        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens

        assert os.path.exists(pickle_ressource_file)
        ressource = cPickle.load(pickle_ressource_file)
        assert type(ressource) == dict
        assert ressource.has_key('grad_E')
        
        # TODO : get r from grad_E
        #        figure out how you want to handle the langevin lambda later on

    def encode(self, X):
        if X.shape[1] != self.n_inputs:
            raise("Using wrong shape[1] for the argument X to DAE.encode. It's %d when it should be %d" % (X.shape[1], self.n_inputs))
        return self.theano_encode(self.Wc, self.c, X)

    def decode(self, H):
        if H.shape[1] != self.n_hiddens:
            raise("Using wrong shape[1] for the argument H to DAE.decode. It's %d when it should be %d" % (H.shape[1], self.n_hiddens))
        return self.theano_decode(self.Wb, self.b, self.s, H)

    def encode_decode(self, X):
        if X.shape[1] != self.n_inputs:
            raise("Using wrong shape[1] for the argument X to DAE.encode_decode. It's %d when it should be %d" % (X.shape[1], self.n_inputs))

        return self.theano_encode_decode(self.Wb, self.Wc, self.b, self.c, self.s, X)

    def model_loss(self, X, noisy_X):
        """
        X:       array-like, shape (n_examples, n_inputs)
        noisy_X: array-like, shape (n_examples, n_inputs)

        Returns  loss: array-like, shape (n_examples,)
        """
        return self.theano_loss(self.Wb, self.Wc, self.b, self.c, self.s, noisy_X, X)

    def jacobian_encode(self, x):
        assert len(x.shape) == 1
        assert x.shape[0] == self.n_inputs
        return self.theano_jacobian_encode(self.Wc, self.c, x)

    def jacobian_encode_decode(self, x):
        assert len(x.shape) == 1
        assert x.shape[0] == self.n_inputs
        return self.theano_jacobian_encode_decode(self.Wb, self.Wc, self.b, self.c, self.s, x)


    def q_read_params(self):
        return DAE_untied_weights.serialize_params_as_q(self.Wb, self.Wc, self.b, self.c, self.s)

    def q_set_params(self, q):
        (self.Wb, self.Wc, self.b, self.c, self.s) = DAE_untied_weights.read_params_from_q(q, self.n_inputs, self.n_hiddens)

    def q_grad(self, q, X, noisy_X):
        (Wb, Wc, b, c, s) = DAE_untied_weights.read_params_from_q(q, self.n_inputs, self.n_hiddens)
        (grad_Wb, grad_Wc, grad_b, grad_c, grad_s) = self.theano_gradients(Wb, Wc, b, c, s, noisy_X, X)

        # There might be a simpler way with theano to do this,
        # but this seems like a good approach.
        if self.want_constant_s:
            grad_s = np.zeros((grad_s.shape))

        return DAE_untied_weights.serialize_params_as_q(grad_Wb, grad_Wc, grad_b, grad_c, grad_s)

    def q_loss(self, q, X, noisy_X):
        (Wb, Wc, b, c, s) = DAE_untied_weights.read_params_from_q(q, self.n_inputs, self.n_hiddens)
        return self.theano_loss(Wb, Wc, b, c, s, noisy_X, X)

def main():
    pass


if __name__ == '__main__':
    main()
