#!/bin/env python

import redis
import numpy as np

r_server = redis.Redis("localhost", 6379)

if not r_server.ping():
    print "Cannot ping server. Exiting."
    quit()


exact_grad_E_from_mixture_mvn_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d10_eig0.1_comp25_001/train_samples_extra.pkl"
#exact_grad_E_from_mixture_mvn_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d2_eig0.1_comp25_001/train_samples_extra.pkl"
script_path = "/u/alaingui/umontreal/denoising_autoencoder/refactor_gp/sampling/sample_dae.py"

experiment_name = "experiment_23_10"

n_samples=500
thinning_factor=10
L_n_E_approx_path = [20]

#L_mcmc_method = ["langevin_grad_E"]
L_mcmc_method = ["MH_langevin_grad_E"]

L_langevin_stddev = [0.1]
L_fake_train_stddev = [0.1, 0.1]
L_langevin_beta = [0.1, 0.05]
L_omit_asymmetric_proposal_factor = [0,1]
#L_proposal_noise_scheme = ['merge_x', 'noise_E', 'noise_r']
L_proposal_noise_scheme = ['merge_x', 'noise_r']

want_overview_plots = "True"
output_dir_counter = 0

for mcmc_method in L_mcmc_method:
    if mcmc_method in ["langevin_grad_E", "MH_langevin_grad_E", "MH_svd_grad_E"]:
        for proposal_noise_scheme in L_proposal_noise_scheme:
            for omit_asymmetric_proposal_factor in L_omit_asymmetric_proposal_factor:
                for fake_train_stddev in L_fake_train_stddev:
                    for langevin_stddev in L_langevin_stddev:
                        for langevin_beta in L_langevin_beta:
                            for n_E_approx_path in L_n_E_approx_path:
                                output_dir = "/data/lisatmp2/alaingui/dae/generated_samples/%s/%0.6d" % (experiment_name, output_dir_counter)
                                output_dir_counter += 1

                                params = (script_path, exact_grad_E_from_mixture_mvn_pickle, fake_train_stddev, n_samples, thinning_factor, n_E_approx_path, langevin_stddev, langevin_beta, want_overview_plots, mcmc_method, omit_asymmetric_proposal_factor, proposal_noise_scheme, output_dir)
                                command = """python %s --exact_grad_E_from_mixture_mvn_pickle=%s --fake_train_stddev=%f --n_samples=%d --thinning_factor=%d --n_E_approx_path=%d --langevin_stddev=%f --langevin_beta=%f --want_overview_plots=%s --mcmc_method=%s --omit_asymmetric_proposal_factor=%d --proposal_noise_scheme=%s --output_dir=%s""" % params

                                print command
                                r_server.rpush("train_gaussian_mixture", command)



