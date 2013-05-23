#!/bin/env python

import cluster
import cluster.configuration
from   cluster.configuration import Configuration
import cluster.database
from   cluster.database import Database
import cluster.job
from   cluster.job import Job

import os, sys
import redis
import numpy as np

r_server = redis.Redis("localhost", 6379)

if not r_server.ping():
    print "Cannot ping server. Exiting."
    quit()


# When you want to use the dataset with d=100, you have to change
# - the pickle file used
# - the output_dir prefix
# - probably stuff like n_hiddens to reflect better the dimensionality


training_script_path = "/u/alaingui/umontreal/denoising_autoencoder/refactor_gp/models/train_dae.py"

d = 10
train_samples_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d%d_eig0.1_comp25_001/train_samples.pkl" % d
valid_samples_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d%d_eig0.1_comp25_001/valid_samples.pkl" % d
test_samples_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d%d_eig0.1_comp25_001/test_samples.pkl" % d

#d = None
#train_samples_pickle = "/data/lisatmp2/alaingui/dae/datasets/mnist/train.pkl"
#valid_samples_pickle = "/data/lisatmp2/alaingui/dae/datasets/mnist/valid.pkl"
#valid_samples_pickle = "/data/lisatmp2/alaingui/dae/datasets/mnist/test.pkl"

config = Configuration()
database = Database(config)


if (d is None) and (train_samples_pickle == "/data/lisatmp2/alaingui/dae/datasets/mnist/train.pkl"):

    L_n_hiddens = [64, 128, 256]
    L_maxiter = [10]
    L_lbfgs_rank = [4]
    #L_act_func = [ '["tanh", "tanh"]', '["tanh", "id"]']
    L_act_func = [ '["tanh", "id"]' ]

    n_reps = 2

    noise_stddevs = [np.exp(s*np.log(10.0)) for s in np.linspace(1,0,5)] + [np.exp(s*np.log(10.0)) for s in np.linspace(0,-2,20)]
    want_early_termination = True

elif d == 10:

    L_n_hiddens = [128]
    L_maxiter = [10000]
    L_lbfgs_rank = [4]
    L_act_func = [ '["tanh", "id"]' ]
    #L_act_func = [ '["tanh", "tanh"]', '["tanh", "id"]']
    #L_act_func = ['["tanh", "tanh"]', '["sigmoid", "tanh"]', '["tanh", "sigmoid"]']

    n_reps = 10

    noise_stddevs = [np.exp(s*np.log(10.0)) for s in np.linspace(1,0,5)] + [np.exp(s*np.log(10.0)) for s in np.linspace(0,-2,20)]
    want_early_termination = True

else:
    quit()

output_dir_counter = 0

for n_hiddens in L_n_hiddens:
    for maxiter in L_maxiter:
        for lbfgs_rank in L_lbfgs_rank:
            for act_func in L_act_func:
                for r in range(n_reps):
                    if (d is None) and (train_samples_pickle == "/data/lisatmp2/alaingui/dae/datasets/mnist/train.pkl"):
                        output_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_mnist/%s/%0.6d" % (experiment_name, output_dir_counter)
                    else:
                        output_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d%d/%s/%0.6d" % (d, experiment_name, output_dir_counter)
                    output_dir_counter += 1

                    params = (training_script_path, n_hiddens, maxiter, lbfgs_rank, act_func, str(noise_stddevs), train_samples_pickle, valid_samples_pickle, str(want_early_termination), output_dir)
                    command = """python %s --n_hiddens=%d --maxiter=%d --lbfgs_rank=%d --act_func='%s' --noise_stddevs='%s' --train_samples_pickle="%s" --valid_samples_pickle="%s" --want_early_termination="%s" --output_dir="%s" """ % params

                    print command
                    r_server.rpush("train_gaussian_mixture", command)


def make_first_calls(training_script_path,
                     train_samples_pickle, 
                     valid_samples_pickle,
                     test_samples_pickle,
                     n_hiddens,
                     lbfgs_rank,
                     act_func,
                     save_hidden_units,
                     maxiter,
                     noise_stddevs,
                     alt_valid_noise_stddevs,
                     output_dir):

    cmd = "%s " % (training_script_path,)
    cmd += " --train_samples_pickle='%s' " % (train_samples_pickle,)
    cmd += " --valid_samples_pickle='%s' " % (valid_samples_pickle,)
    cmd += " --test_samples_pickle='%s' "  % (test_samples_pickle,)

    cmd += " --n_hiddens=%d "  % (n_hiddens,)
    cmd += " --lbfgs_rank=%d "  % (lbfgs_rank,)
    cmd += " --act_func='%s' "  % (act_func,)
    cmd += " --save_hidden_units=%d "  % (save_hidden_units,)
    cmd += " --maxiter=%d "  % (maxiter,)

    cmd += " --noise_stddevs='%s' "  % (str(noise_stddevs),)
    cmd += " --alt_valid_noise_stddevs='%s' "  % (str(alt_valid_noise_stddevs),)

    cmd += " --output_dir='%s' "  % (output_dir,)
