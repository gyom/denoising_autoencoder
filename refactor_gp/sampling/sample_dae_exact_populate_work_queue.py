#!/bin/env python

import redis
import numpy as np

r_server = redis.Redis("localhost", 6379)

if not r_server.ping():
    print "Cannot ping server. Exiting."
    quit()


exact_grad_E_from_mixture_mvn_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d2_eig0.1_comp25_001/train_samples_extra.pkl"
script_path = "/u/alaingui/umontreal/denoising_autoencoder/refactor_gp/sampling/sample_dae.py"

experiment_name = "experiment_14"

n_samples=500
thinning_factor=1000
L_n_E_approx_path = [5]

L_mcmc_method = ["langevin_grad_E", "MH_langevin_grad_E"]

L_langevin_stddev = [0.1, 0.05, 0.01]
L_fake_train_stddev = [0.1, 0.05, 0.01]
L_langevin_beta = [1.0, 0.5, 0.1]

want_overview_plots = "True"
output_dir_counter = 0

for mcmc_method in L_mcmc_method:
    if mcmc_method in ["langevin_grad_E", "MH_langevin_grad_E", "MH_svd_grad_E"]:
        for fake_train_stddev in L_fake_train_stddev:
            for langevin_stddev in L_langevin_stddev:
                for langevin_beta in L_langevin_beta:
                    for n_E_approx_path in L_n_E_approx_path:
                        output_dir = "/data/lisatmp2/alaingui/dae/generated_samples/%s/%0.6d" % (experiment_name, output_dir_counter)
                        output_dir_counter += 1

                        params = (script_path, exact_grad_E_from_mixture_mvn_pickle, fake_train_stddev, n_samples, thinning_factor, n_E_approx_path, langevin_stddev, langevin_beta, want_overview_plots, mcmc_method, output_dir)
                        command = """python %s --exact_grad_E_from_mixture_mvn_pickle=%s --fake_train_stddev=%f --n_samples=%d --thinning_factor=%d --n_E_approx_path=%d --langevin_stddev=%f --langevin_beta=%f --want_overview_plots=%s --mcmc_method=%s --output_dir=%s""" % params

                        print command
                        r_server.rpush("train_gaussian_mixture", command)



