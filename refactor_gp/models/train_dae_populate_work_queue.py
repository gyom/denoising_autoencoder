#!/bin/env python

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

d = 10
training_script_path = "/u/alaingui/umontreal/denoising_autoencoder/refactor_gp/train_dae.py"
train_samples_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d%d_eig0.1_comp25_001/train_samples.pkl" % d

if d == 10:
    #L_n_hiddens = [10,25]
    L_n_hiddens = [20,30,50,80,100]
    L_maxiter = [100,1000]
    L_lbfgs_rank = [1,4]
    L_act_func = [ '["tanh", "tanh"]', '["tanh", "id"]']
    #L_act_func = ['["tanh", "tanh"]', '["sigmoid", "tanh"]', '["tanh", "sigmoid"]']

    noise_stddevs = [np.exp(s*np.log(10.0)) for s in np.linspace(0,-5,10)]
    # noise_stddevs = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001]
    want_early_termination = True

elif d == 100:
    #L_n_hiddens = [1,2,5,10,20,50]
    L_n_hiddens = [20,500,1000]
    L_maxiter = [10000]
    L_lbfgs_rank = [1, 5]
    L_act_func = [ '["tanh", "tanh"]' ]
    noise_stddevs = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001]

    want_early_termination = True
else:
    quit()

output_dir_counter = 0

for n_hiddens in L_n_hiddens:
    for maxiter in L_maxiter:
        for lbfgs_rank in L_lbfgs_rank:
            for act_func in L_act_func:
                
                output_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d%d/experiment_04/%0.6d" % (d, output_dir_counter)
                output_dir_counter += 1

                params = (training_script_path, n_hiddens, maxiter, lbfgs_rank, act_func, str(noise_stddevs), train_samples_pickle, str(want_early_termination), output_dir)
                command = """python %s --n_hiddens=%d --maxiter=%d --lbfgs_rank=%d --act_func='%s' --noise_stddevs='%s' --train_samples_pickle="%s" --want_early_termination="%s" --output_dir="%s" """ % params

                print command
                r_server.rpush("train_gaussian_mixture", command)
