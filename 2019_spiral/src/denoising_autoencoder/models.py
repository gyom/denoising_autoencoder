from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# import tensorflow.contrib.slim as slim

class ModelMLP(object):
    """
    An instance of this class contains everything with the model being applied
    to an input tensor. It features more than just the weights of the model.
    Multiple applications are handled through scope sharing.
    """
    def __init__(self, inputs, layer_dims, test_phase=False, output_dim=None):
        """
        We really expect this constructor call to be made within a proper scope
        so that variable reuse is properly used.
        """
        self.inputs = inputs  # a tensor
        self.layer_dims = layer_dims  # a list
        self.test_phase = test_phase  # a bool
        if output_dim is not None:
            self.output_dim = output_dim
        else:
            self.output_dim = inputs.get_shape()[-1]
        self._build_model()

    def _build_model(self):
        """
        It does not want arguments because it is going to pick whatever it needs.
        """

        start_trainable_variables = tf.trainable_variables()
        net = self.inputs

        for layer_dim in layer_dims:

            net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.leaky_relu)

            # Never dropout for first or last layer.
            if not self.test_phase:
                net = tf.dropout(net, 0.5)

        # last layer back to the original input dimension
        self.outputs = tf.layers.dense(inputs=net, units=self.output_dim, activation=tf.identity)

        end_trainable_variables = tf.trainable_variables()
        self.L_params = [ param for param in end_trainable_variables if param not in start_trainable_variables ]


class ModelXPlusSigma2MLP(ModelMLP):
    """
    r(x) = x + sigma^2 * neural_network(x)
    and we hope that neural_network(x) will end up as d/dx log p(x)
    """

    def __init__(self, inputs, layer_dims, sigma2, test_phase=False):
        """
        `sigma2` is sigma squared
        """
        super(ModelXPlusSigma2MLP, self).__init__(inputs, layer_dims, test_phase)
        self.outputs = inputs + sigma2 * self.outputs


# class ModelXPlusSigma2derivativeMLP(ModelMLP):
#     """
#     r(x) = x + sigma^2 * d/dx neural_network(x)
#     and we hope that neural_network(x) will end up as log p(x)
#     """
#     def __init__(self, inputs, layer_dims, sigma2, test_phase=False):
#         """
#         `sigma2` is sigma squared
#         """
#         super(ModelXPlusSigma2derivativeMLP, self).__init__(inputs, layer_dims, test_phase, output_dim=1)
#
#         # THIS DOES NOT WORK VERY NICELY BECAUSE OF THE BATCH THING.
#         # THINK ABOUT THIS FURTHER.
#         net = self.outputs
#         tf.grad(net, )
#
#         self.outputs = inputs + sigma2 * self.outputs


##### This function is rather generic. #####
##### Not specific to this model.      #####

def get_training_op(loss, grads, L_params, hparams, global_step, optimizer_suffix=""):
    """Obtain a training operator for all the probes involved in a model.

    Args :
        loss : The loss to minimize.
        grads : The gradients associated to each params, in the same order.
        L_params : The parameters to optimize.
        hparams : The usual full set of hyperparameters from which we will select
                  only what we really want to use. Same as with everywhere.
        global_step : The usual tensorflow global step.

        Either `loss` or `grads` has to be `None`.
        We can use either, but not both.

    Returns :
        A training operator to be called in a `session.run`.
    """

    assert loss is None or grads is None
    assert loss is not None or grads is not None

    learning_rate = tf.train.exponential_decay(
        hparams.learning_rate,  # Base learning rate.
        global_step * hparams.batch_size,
        decay_steps=hparams.learning_rate_decay_steps,
        decay_rate=hparams.learning_rate_decay_rate,
        staircase=False)

    #tf.summary.scalar("learning rate", learning_rate)
    #grads, _ = tf.clip_by_global_norm(
    #              tf.gradients(loss, L_params), hparams.max_grad_norm)
    if grads is None:
        grads = tf.gradients(loss, L_params)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.variable_scope("RMSPropOptimizer" + optimizer_suffix, reuse=None):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=hparams.rmsprop_decay,
            momentum=hparams.rmsprop_momentum)
        train_op = optimizer.apply_gradients(
            zip(grads, L_params), global_step=global_step)
        return (train_op, learning_rate)




"""
"""
