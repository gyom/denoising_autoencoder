from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.insert(0,'/current_project/src')
import denoising_autoencoders
import denoising_autoencoders.spiral

import numpy as np
import tensorflow as tf

import itertools

# This is inspired by
#     https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator

def get_spiral_dataset_iterator(batch_size, **kwargs):
    def gen():
        for i in itertools.count(1):
            data = denoising_autoencoders.spiral.sample( N=batch_size, **kwargs)
            assert data.shape == (batch_size, 2)
            yield data.astype(np.float32)

    ds = Dataset.from_generator(
        gen, tf.float32, tf.TensorShape([batch_size, 2]))
    value = ds.make_one_shot_iterator().get_next()
    return value
