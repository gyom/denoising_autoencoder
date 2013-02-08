

import numpy as np

import cPickle
mnist_dataset = cPickle.load(open('/data/lisa/data/mnist/mnist.pkl', 'rb'))
mnist_train = mnist_dataset[0]
mnist_train_data, mnist_train_labels = mnist_train


from dae_untied_weights import DAE_untied_weights

mydae = DAE_untied_weights(n_inputs = 784,
                           n_hiddens = 1024,
                           act_func = ['tanh', 'sigmoid'])

mydae.fit_with_decreasing_noise(mnist_train_data[0:2000,:],
                                [0.1, 0.05, 0.01, 0.001],
                                {'method' : 'fmin_cg',
                                 'maxiter' : 500,
                                 'gtol':0.001})


mydae.fit_with_decreasing_noise(mnist_train_data[0:2000,:],
                                [0.1, 0.05, 0.01, 0.001],
                                {'method' : 'fmin_bfgs',
                                 'maxiter' : 500})


cPickle.dump({'Wc':mydae.Wc, 'Wb':mydae.Wb, 'b':mydae.b, 'c':mydae.c, 's':mydae.s, 'act_func':mydae.act_func},
             open("/u/alaingui/umontreal/denoising_autoencoder/mcmc_pof/trained_models/mydae_2013_02_07.pkl", "w"))
