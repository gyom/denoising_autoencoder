

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



mnist_dataset = cPickle.load(open("/data/lisa/data/mnist/mnist.pkl", "r"))
Ntrain = 500
Xtrain = mnist_dataset[0][0][0:Ntrain,:]

H0 = encode_x_to_h0( Xtrain )
H1 = encode_h0_to_h1( H0 )

recH0 = decode_h1_to_h0( H1 )
recX  = decode_h0_to_x( recH0 )

losses = ((Xtrain - recX)**2).sum(axis=1)

I = refactor_gp.yann_dauphin_utils.tile_raster_images(recX[0:100,:], (28,28), (10,10))


import matplotlib
import pylab
import matplotlib.pyplot as plt

from PIL import Image
im = Image.fromarray(I)
im.save("/u/alaingui/umontreal/denoising_autoencoder/refactor_gp/junk/mnist.png")




######## H1 <---> H2 ########
#
#def encode_h1_to_h2( H1 ):
#    return sigmoid( H1.dot( mnist_cae_trained_parameters['W20'] ) + mnist_cae_trained_parameters['b20'] )
#
#def decode_h2_to_h1( H2 ):
#    return sigmoid( H2.dot( mnist_cae_trained_parameters['W20'].T ) + mnist_cae_trained_parameters['b_prime20'] )
#
######## H2 <---> H3 ########
#
#def encode_h2_to_h3( H2 ):
#    return sigmoid( H2.dot( mnist_cae_trained_parameters['W30'] ) + mnist_cae_trained_parameters['b30'] )
#
#def decode_h3_to_h2( H3 ):
#    return sigmoid( H3.dot( mnist_cae_trained_parameters['W30'].T ) + mnist_cae_trained_parameters['b_prime30'] )


#params0 = cPickle.load(open(os.path.join(cae_dir, "rbm_yann_mnist_3layers/params_0.pkl"), "r"))
#params1 = cPickle.load(open(os.path.join(cae_dir, "rbm_yann_mnist_3layers/params_1.pkl"), "r"))
#params2 = cPickle.load(open(os.path.join(cae_dir, "rbm_yann_mnist_3layers/params_2.pkl"), "r"))

#for params in [params0, params1, params2]:
#    for e in params:
#        print e.shape