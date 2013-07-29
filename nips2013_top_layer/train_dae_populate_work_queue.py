#!/bin/env python

import redis
import numpy as np

#r_server = redis.Redis("localhost", 6379)
r_server = redis.Redis("eos1", 6379)

if not r_server.ping():
    print "Cannot ping server. Exiting."
    quit()


# When you want to use the dataset with d=100, you have to change
# - the pickle file used
# - the output_dir prefix
# - probably stuff like n_hiddens to reflect better the dimensionality


training_script_path = "/u/alaingui/umontreal/denoising_autoencoder/refactor_gp/models/train_dae.py"

d = None
debug = False
if debug:
    train_samples_pickle = "/data/lisatmp2/alaingui/mnist/yann/yann_train_H1_trim_100.pkl"
    valid_samples_pickle = "/data/lisatmp2/alaingui/mnist/yann/yann_valid_H1_trim_100.pkl"
else:
    #train_samples_pickle = "/data/lisatmp2/alaingui/mnist/yann/yann_train_H1_trim_10000.pkl"
    train_samples_pickle = "/data/lisatmp2/alaingui/mnist/yann/yann_train_H1.pkl"
    valid_samples_pickle = "/data/lisatmp2/alaingui/mnist/yann/yann_valid_H1.pkl"

#experiment_name = "experiment_05_yann_mnist_H1_kicking"
#experiment_name = "experiment_05_yann_mnist_H1_walkback"

if True:

    L_n_hiddens = [128, 256, 512]
    L_maxiter = [20]
    L_lbfgs_rank = [8]
    #L_act_func = [ '["tanh", "sigmoid"]', '["sigmoid", "sigmoid"]']
    L_act_func = [ '["sigmoid", "sigmoid"]']
    n_reps = 4
    loss_function_desc = "cross-entropy"
    want_constant_s = "True"

    mode = 1

    kicking_param_p = 0.5
    walkback_param_p = 0.5
    if mode == 1:
        experiment_name = "experiment_09_yann_mnist_H1_normal"
        S      = [np.exp(s*np.log(10.0)) for s in np.linspace(1,0,7)] + [np.exp(s*np.log(10.0)) for s in np.linspace(0,-2,13)]
        noise_stddevs = {}
        noise_stddevs['train'] = [{'target':s, 'sampled':s} for s in S]
        noise_stddevs['valid'] = [{'target':s, 'sampled':s} for s in S]
        noise_stddevs['valid_kicking'] = [{'sampled':s, 'kicking':10*s, 'kicking_param_p':kicking_param_p} for s in S]
        noise_stddevs['valid_walkback'] = [{'sampled':s, 'walkback_param_p':walkback_param_p} for s in S]
        noise_stddevs['gentle_valid'] = [{'target':10*s, 'sampled':10*s} for s in S]
    #elif mode == 2:
    #    experiment_name = "experiment_09_yann_mnist_H1_kicking"
    #    S      = [np.exp(s*np.log(10.0)) for s in np.linspace(1,0,7)] + [np.exp(s*np.log(10.0)) for s in np.linspace(0,-2,13)]
    #    noise_stddevs = {}
    #    noise_stddevs['train'] = [{'sampled':s, 'kicking':10*s, 'kicking_param_p':kicking_param_p} for s in S]
    #    noise_stddevs['valid'] = [{'target':s, 'sampled':s} for s in S]
    #    noise_stddevs['valid_kicking'] = [{'sampled':s, 'kicking':10*s, 'kicking_param_p':kicking_param_p} for s in S]
    #    noise_stddevs['valid_walkback'] = [{'sampled':s, 'walkback_param_p':walkback_param_p} for s in S]
    #    noise_stddevs['gentle_valid'] = [{'target':10*s, 'sampled':10*s} for s in S]
    elif mode == 3:
        experiment_name = "experiment_09_yann_mnist_H1_walkback"
        S      = [np.exp(s*np.log(10.0)) for s in np.linspace(1,0,7)] + [np.exp(s*np.log(10.0)) for s in np.linspace(0,-2,13)]
        noise_stddevs = {}
        noise_stddevs['train'] = [{'sampled':s, 'walkback_param_p':walkback_param_p} for s in S]
        noise_stddevs['valid'] = [{'target':s, 'sampled':s} for s in S]
        noise_stddevs['valid_kicking'] = [{'sampled':s, 'kicking':10*s, 'kicking_param_p':kicking_param_p} for s in S]
        noise_stddevs['valid_walkback'] = [{'sampled':s, 'walkback_param_p':walkback_param_p} for s in S]
        noise_stddevs['gentle_valid'] = [{'target':10*s, 'sampled':10*s} for s in S]

else:
    quit()

output_dir_counter = 0

for maxiter in L_maxiter:
    for n_hiddens in L_n_hiddens:
        for lbfgs_rank in L_lbfgs_rank:
            for act_func in L_act_func:
                for r in range(n_reps):
                    output_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/mnist_yann_H1/%s/%0.6d" % (experiment_name, output_dir_counter)
                    output_dir_counter += 1

                    params = (training_script_path, n_hiddens, maxiter, lbfgs_rank, act_func, want_constant_s, loss_function_desc, str(noise_stddevs).replace("'", '"'), train_samples_pickle, valid_samples_pickle, output_dir)
                    command = """python %s --n_hiddens=%d --maxiter=%d --lbfgs_rank=%d --act_func='%s' --want_constant_s=%s --loss_function_desc=%s --noise_stddevs='%s' --train_samples_pickle="%s" --valid_samples_pickle="%s" --output_dir="%s" """ % params

                    print command
                    print
                    r_server.rpush("train_yann_mnist", command)
