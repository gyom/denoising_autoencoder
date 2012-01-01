
import cPickle
import os
import numpy as np

cae_dir = "/data/lisa/exp/mesnilgr/shared/"
mnist_cae_trained_parameters = cPickle.load(open(os.path.join(cae_dir, "cae_salah_mnist.cp"), "r"))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

######## X <---> H0 ########
#
def encode_x_to_h0( X ):
    return sigmoid( X.dot( mnist_cae_trained_parameters['W00'] ) + mnist_cae_trained_parameters['b00'] )

def decode_h0_to_x( H0 ):
    return sigmoid( H0.dot( mnist_cae_trained_parameters['W00'].T ) + mnist_cae_trained_parameters['b_prime00'] )


######## H0 <---> H1 ########
#
def encode_h0_to_h1( H0 ):
    return sigmoid( H0.dot( mnist_cae_trained_parameters['W10'] ) + mnist_cae_trained_parameters['b10'] )

def decode_h1_to_h0( H1 ):
    return sigmoid( H1.dot( mnist_cae_trained_parameters['W10'].T ) + mnist_cae_trained_parameters['b_prime10'] )
