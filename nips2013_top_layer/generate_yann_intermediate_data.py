
import numpy as np
import cPickle
import os

import nips2013_top_layer
import nips2013_top_layer.common
from nips2013_top_layer.common import encode_h0_to_h1, encode_x_to_h0, decode_h1_to_h0, decode_h0_to_x 

mnist_dataset = cPickle.load(open("/data/lisa/data/mnist/mnist.pkl", "r"))

output_dir = "/data/lisatmp2/alaingui/mnist/yann"

mean_losses = {}
for phase in ['train', 'valid', 'test']:

    if phase == 'train':
        X = mnist_dataset[0][0]
        #labels = mnist_dataset[0][1]
    elif phase == 'valid':
        X = mnist_dataset[1][0]
        #labels = mnist_dataset[1][1]
    elif phase == 'test':
        X = mnist_dataset[2][0]
        #labels = mnist_dataset[2][1]

    H0 = encode_x_to_h0( X )
    H1 = encode_h0_to_h1( H0 )

    rH0 = decode_h1_to_h0( H1 )
    rX  = decode_h0_to_x( rH0 )

    quadratic_losses = ((X - rX)**2).sum(axis=1)
    cross_entropy_losses = (X * np.log(rX) + (1-X) * np.log(1-rX)).sum(axis=1)

    mean_losses[phase] = {}
    mean_losses[phase]['quadratic_losses'] = quadratic_losses.mean()
    mean_losses[phase]['cross_entropy_losses'] = cross_entropy_losses.mean()

    print "phase %s" % (phase,)
    print "    mean %s : %f" % ('quadratic loss', mean_losses[phase]['quadratic_losses'])
    print "    mean %s : %f" % ('cross-entropy loss', mean_losses[phase]['cross_entropy_losses'])

    cPickle.dump(X, open(os.path.join(output_dir, "yann_%s_X.pkl" % (phase,)), "w"))
    cPickle.dump(H0, open(os.path.join(output_dir, "yann_%s_H0.pkl" % (phase,)), "w"))
    cPickle.dump(H1, open(os.path.join(output_dir, "yann_%s_H1.pkl" % (phase,)), "w"))
    cPickle.dump(rH0, open(os.path.join(output_dir, "yann_%s_rH0.pkl" % (phase,)), "w"))
    cPickle.dump(rX, open(os.path.join(output_dir, "yann_%s_rX.pkl" % (phase,)), "w"))

    #I = refactor_gp.yann_dauphin_utils.tile_raster_images(recX[0:100,:], (28,28), (10,10))

import json
json.dump(mean_losses, open(os.path.join(output_dir, "mean_losses.json"), "w"), sort_keys=False, indent=4, separators=(',', ': '))
cPickle.dump(mean_losses, open(os.path.join(output_dir, "mean_losses.pkl"), "w"))

