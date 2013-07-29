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

import refactor_gp
import refactor_gp.models
from   refactor_gp.models import dae

from dae import DAE

class DAE_untied_weights(DAE):
    """
    A DAE with tanh input units and tanh hidden units.
    """
    def __init__(self,
                 n_inputs=None,
                 n_hiddens=None,
                 Wc=None, Wb=None,
                 c=None,  b=None,
                 s=None, act_func=['tanh', 'tanh'],
                 want_constant_s = False,
                 loss_function_desc = None,
                 dae_pickle_file=None):
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
        s : array-like, shape (n_inputs,), optional
            Applied after the second tanh at the output.
            Allows us to represent values in a range [-4,4]
            instead of just [-1,1] by using alpha = 4.0.
            This used to be just a real number, but now
            we use a vector because we want to scale each output
            independantly.
        loss_function_desc : either "quadratic" (default) or "cross-entropy"
        dae_pickle_file : pickle file previously saved from this implementation
        """

        # Note that we should probably be using some scheme
        # where we have two constructors. One takes arguments,
        # and the other takes a pickle file and feeds the
        # parameters into the first constructor.
        # This would avoid potential issues with saving/loading
        # trained DAEs.
        if dae_pickle_file:
            self.load_pickle(dae_pickle_file)
            return

        # These values are to be treated as READ-ONLY,
        # but they're actually modified when we load the
        # DAE from a pickle file. This infraction is acceptable
        # because the DAE is not used in any way before
        # we load the pickle and set the values for n_inputs and n_hiddens.
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens

        # These values are expected to be modified by
        # algorithms that take the DAE instance as parameter.
        # ex : any training function
        self.reset_params()
        if not (Wc == None):
            self.Wc = Wc
        if not (Wb == None):
            self.Wb = Wb
        if not (c == None):
            self.c = c
        if not (b == None):
            self.b = b
        if not (s == None):
            self.s = s

        self.want_constant_s = want_constant_s
        self.loss_function_desc = loss_function_desc

        if len(act_func) != 2:
            raise("Need to specify two activation functions from : ['tanh', 'sigmoid', 'id'].")
        else:
            for f in act_func:
                if not f in ['tanh', 'sigmoid', 'id']:
                    raise("Unrecognized activation function. Should be from : ['tanh', 'sigmoid', 'id'].")
            if act_func[0] == 'id':
                print "It's a bad idea to use the identity as first activation function. \nMaybe you got the ordering mixed up ?"
            if act_func[1] == 'id':
                print "To use the identity function as second activation function, we will keep s=[1.0, ..., 1.0] constant. This avoids clashes with Wb."
                self.want_constant_s = True
        self.act_func = act_func

        self.want_plus_x = False
        self.tied_weights = False

        # then setup the theano functions once
        self.theano_setup()
        self.theano_setup_flat()
    
    def theano_setup(self):
    
        if self.loss_function_desc is None:
            self.loss_function_desc = "quadratic"
        else:
            assert self.loss_function_desc in ["quadratic", "cross-entropy"]

        if self.loss_function_desc == "cross-entropy":
            # It would be possible to change this automatically,
            # but I think we would not be doing the user a service
            # by not pointing out this configuration problem.
            # It's better to issue an error here and let the user
            # fix the configuration higher up.
            if self.want_constant_s == False:
                print "You cannot use the cross-entropy loss and not ask for the scaling constant 's' to be optimized."
                print "It has to be 1.0. Quitting here."
                quit()
            elif not (self.act_func[1] == 'sigmoid'):
                print "You cannot use the cross-entropy loss and not use a sigmoid as second activation function."
                print "Quitting here."
                quit()

        # The matrices Wb and Wc were originally tied.
        # Because of that, I decided to keep Wb and Wc with
        # the same shape (instead of being transposed) to
        # avoid disturbing the code as much as possible.

        Wb = T.matrix('Wb')
        Wc = T.matrix('Wc')
        b = T.vector('b')
        c = T.vector('c')
        s = T.vector('s')
        x = T.matrix('x')
    
        h_act = T.dot(x, Wc) + c
        if self.act_func[0] == 'tanh':
            h = T.tanh(h_act)
        elif self.act_func[0] == 'sigmoid':
            h = T.nnet.sigmoid(h_act)
        elif self.act_func[0] == 'id':
            h = T.nnet.sigmoid(h_act)
            # bad idea
            h = h_act
        else:
            raise("Invalid act_func[0]")

        r_act = T.dot(h, Wb.T) + b
        if self.act_func[1] == 'tanh':
            r = s * T.tanh(r_act)
        elif self.act_func[1] == 'sigmoid':
            r = s * T.nnet.sigmoid(r_act)
        elif self.act_func[1] == 'id':
            r = s * r_act
        else:
            raise("Invalid act_func[1]")

        importance_sampling_weights = T.vector('importance_sampling_weights')

        # Another variable to be able to call a function
        # with a noisy x and compare it to a reference x.
        y = T.matrix('y')

        # Make importance_sampling_weights have a broadcastable dimension
        # to multiply all the reconstructed components.
        # This is useful if we are minimizing the loss function
        # using a distribution that is slightly more noisy and
        # we need to have some correction factor to the loss
        # to make it equivalent.

        if self.loss_function_desc == "quadratic":
            loss = ((r - y)**2) * importance_sampling_weights[:,np.newaxis]
        else:
            loss = -((1-y)*T.log(1-r) + y*T.log(r)) * importance_sampling_weights[:,np.newaxis]

        sum_loss = T.sum(loss)
        
        # theano_encode_decode : vectorial function in argument X.
        # theano_loss : vectorial function in argument X.
        # theano_gradients : returns triplet of gradients, each of
        #                    which involves the all data X summed
        #                    so it's not a "vectorial" function.

        self.theano_encode_decode = function([Wb,Wc,b,c,s,x], r, allow_input_downcast=True)
        self.theano_loss = function([Wb,Wc,b,c,s,x,y,importance_sampling_weights], loss, allow_input_downcast=True)

        self.theano_gradients = function([Wb,Wc,b,c,s,x,y,importance_sampling_weights],
                                         [T.grad(sum_loss, Wb), T.grad(sum_loss, Wc),
                                          T.grad(sum_loss, b),  T.grad(sum_loss, c),
                                          T.grad(sum_loss, s)], allow_input_downcast=True)
        # other useful theano functions for the experiments that involve
        # adding noise to the hidden states
        self.theano_encode = function([Wc,c,x], h, allow_input_downcast=True)
        self.theano_decode = function([Wb,b,s,h], r, allow_input_downcast=True)


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

    def model_loss(self, X, noisy_X, importance_sampling_weights = None):
        """
        X:       array, shape (n_examples, n_inputs)
        noisy_X: array, shape (n_examples, n_inputs)
        importance_sampling_weights : optional, array, shape (n_examples,)

        Returns  loss: array-like, shape (n_examples,)
        """

        if importance_sampling_weights is None:
            importance_sampling_weights = np.ones((X.shape[0],))

        return self.theano_loss(self.Wb, self.Wc, self.b, self.c, self.s, noisy_X, X, importance_sampling_weights)



    def theano_setup_flat(self):
    
        Wb = T.matrix('Wb')
        Wc = T.matrix('Wc')
        b = T.vector('b')
        c = T.vector('c')
        s = T.vector('s')
        x = T.vector('x')
    
        h_act = T.dot(x, Wc) + c
        if self.act_func[0] == 'tanh':
            h = T.tanh(h_act)
        elif self.act_func[0] == 'sigmoid':
            h = T.nnet.sigmoid(h_act)
        elif self.act_func[0] == 'id':
            h = h_act
        else:
            raise("Invalid act_func[0]")

        r_act = T.dot(h, Wb.T) + b
        if self.act_func[1] == 'tanh':
            r = s * T.tanh(r_act)
        elif self.act_func[1] == 'sigmoid':
            r = s * T.nnet.sigmoid(r_act)
        elif self.act_func[1] == 'id':
            r = s * r_act
        else:
            raise("Invalid act_func[1]")

        # Finally, we are interested in the derivatives of the
        # encode and encode_decode functions because these are
        # useful in the context of sampling.
        #
        # For now I'm not sure if it's even possible to do this
        # because X is defined to be a matrix.
        # That's again thanks to the dual-mission of vectoring
        # that certain dimensions are given.
        self.theano_jacobian_encode = function([Wc,c,x], theano.gradient.jacobian(h,x), allow_input_downcast=True)
        self.theano_jacobian_encode_decode = function([Wb,Wc,b,c,s,x], theano.gradient.jacobian(r,x), allow_input_downcast=True)

    def jacobian_encode(self, x):
        assert len(x.shape) == 1
        assert x.shape[0] == self.n_inputs
        return self.theano_jacobian_encode(self.Wc, self.c, x)

    def jacobian_encode_decode(self, x):
        assert len(x.shape) == 1
        assert x.shape[0] == self.n_inputs
        return self.theano_jacobian_encode_decode(self.Wb, self.Wc, self.b, self.c, self.s, x)



    def reset_params(self):

        #(scale_Wb, scale_Wc) = (1.0, 1.0)

        # Scaling them in a way that anticipates the fact that
        # with random inputs centered around 0.0, we should want
        # to end up with values distributed according to roughly N(0,1).
        scale_Wb = 1.0 / np.sqrt(self.n_inputs)
        scale_Wc = 1.0 / np.sqrt(self.n_hiddens)

        self.Wb = np.random.uniform( low = -1.0, high = 1.0, size=(self.n_inputs, self.n_hiddens) ) * scale_Wb
        self.Wc = np.random.uniform( low = -1.0, high = 1.0, size=(self.n_inputs, self.n_hiddens) ) * scale_Wc
        self.b  = np.random.uniform( low = -0.1, high = 0.1, size=(self.n_inputs,) )
        self.c  = np.random.uniform( low = -0.1, high = 0.1, size=(self.n_hiddens,) )
        # s is a factor and 1 is the neutral element for the product, so we'll start with values close to 1.
        self.s  = np.ones((self.n_inputs,))
        #self.s  = 1.0

    
    # tip : Write the U(q) and grad_U(q) methods
    #       first to figure out what you'll need to implement here.
    #       Otherwise you can't design the interface here beforehand.

    def q_read_params(self):
        return DAE_untied_weights.serialize_params_as_q(self.Wb, self.Wc, self.b, self.c, self.s)

    def q_set_params(self, q):
        (self.Wb, self.Wc, self.b, self.c, self.s) = DAE_untied_weights.read_params_from_q(q, self.n_inputs, self.n_hiddens)

    def q_grad(self, q, X, noisy_X, importance_sampling_weights = None):
        (Wb, Wc, b, c, s) = DAE_untied_weights.read_params_from_q(q, self.n_inputs, self.n_hiddens)

        if importance_sampling_weights is None:
            importance_sampling_weights = np.ones((X.shape[0],), dtype=np.float32)

        (grad_Wb, grad_Wc, grad_b, grad_c, grad_s) = self.theano_gradients(Wb, Wc, b, c, s, noisy_X, X, importance_sampling_weights)

        # There might be a simpler way with theano to do this,
        # but this seems like a good approach.
        if self.want_constant_s:
            grad_s = np.zeros((grad_s.shape), dtype=np.float32)

        return DAE_untied_weights.serialize_params_as_q(grad_Wb, grad_Wc, grad_b, grad_c, grad_s)

    def q_loss(self, q, X, noisy_X, importance_sampling_weights):
        (Wb, Wc, b, c, s) = DAE_untied_weights.read_params_from_q(q, self.n_inputs, self.n_hiddens)

        if importance_sampling_weights is None:
            importance_sampling_weights = np.ones((X.shape[0],), dtype=np.float32)

        return self.theano_loss(Wb, Wc, b, c, s, noisy_X, X, importance_sampling_weights)



    @staticmethod
    def serialize_params_as_q(Wb, Wc, b, c, s):
        return np.hstack((Wb.reshape((-1,)),
                          Wc.reshape((-1,)),
                          b.reshape((-1,)),
                          c.reshape((-1,)),
                          s.reshape((-1,))     )).reshape((-1,))

    @staticmethod
    def read_params_from_q(q, n_inputs, n_hiddens):

        n_elems_Wb = n_inputs * n_hiddens
        n_elems_Wc = n_inputs * n_hiddens
        n_elems_b = n_inputs
        n_elems_c = n_hiddens
        n_elems_s = n_inputs

        bounds = (0,
                  n_elems_Wb,
                  n_elems_Wb + n_elems_Wc,
                  n_elems_Wb + n_elems_Wc + n_elems_b,
                  n_elems_Wb + n_elems_Wc + n_elems_b + n_elems_c,
                  n_elems_Wb + n_elems_Wc + n_elems_b + n_elems_c + n_elems_s)

        Wb = q[ bounds[0] : bounds[1] ].reshape((n_inputs, n_hiddens)).copy()
        Wc = q[ bounds[1] : bounds[2] ].reshape((n_inputs, n_hiddens)).copy()
        b  = q[ bounds[2] : bounds[3] ].reshape((n_elems_b,)).copy()
        c  = q[ bounds[3] : bounds[4] ].reshape((n_elems_c,)).copy()
        s  = q[ bounds[4] : bounds[5] ].reshape((n_elems_s,)).copy()

        return (Wb, Wc, b, c, s)


    def save_pickle(self, pickle_file_path):
        import cPickle
        cPickle.dump({'q':self.q_read_params(),
                      'act_func':self.act_func,
                      'n_inputs':self.n_inputs,
                      'n_hiddens':self.n_hiddens,
                      'want_constant_s':self.want_constant_s,
                      'loss_function_desc':self.loss_function_desc},
                     open(pickle_file_path, "w"))

    def load_pickle(self, pickle_file_path):
        import cPickle
        assert os.path.exists(pickle_file_path)
        params = cPickle.load(open(pickle_file_path, "r"))
        assert params.has_key('q')
        assert params.has_key('act_func')
        assert params.has_key('n_inputs')
        assert params.has_key('n_hiddens')

        # wipe out everything that was in the
        # class before we changed it to a
        # new configuration
        self.n_inputs = params['n_inputs']
        self.n_hiddens = params['n_hiddens']
        self.act_func = params['act_func']
        self.q_set_params(params['q'])
        if params.has_key('want_constant_s'):
            self.want_constant_s = params['want_constant_s']
        if params.has_key('loss_function_desc'):
            self.loss_function_desc = params['loss_function_desc']
        # we need to regenerate the theano functions
        self.theano_setup()
        self.theano_setup_flat()


def main():
    pass


if __name__ == '__main__':
    main()
