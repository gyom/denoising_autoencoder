#!/bin/env python

import redis
import numpy as np

r_server = redis.Redis("localhost", 6379)

if not r_server.ping():
    print "Cannot ping server. Exiting."
    quit()


# ./sample_dae.py --pickled_dae_dir=/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d10/experiment_07/000021 --n_samples=500 --thinning_factor=10 --mcmc_method=langevin_grad_E --n_E_approx_path=10 --want_overview_plots=True --output_dir=${HOME}/Documents/tmp/07_000021_langevin_grad_E_2



script_path = "/u/alaingui/umontreal/denoising_autoencoder/refactor_gp/sampling/sample_dae.py"
# selected manually
pickled_dae_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d10/experiment_07/000021"

n_samples=500
thinning_factor=1000
L_n_E_approx_path = [0,5,20]
L_mcmc_method = ["langevin_grad_E", "MH_langevin_grad_E", "MH_svd_grad_E"]

L_langevin_lambda = [0.0001, 0.000001]
L_proposal_stddev = [0.1, 0.01, 0.001]

want_overview_plots = "True"


output_dir_counter = 0
experiment_name = "experiment_03"

for n_E_approx_path in L_n_E_approx_path:
    for mcmc_method in L_mcmc_method:

        if mcmc_method in ["langevin_grad_E", "MH_langevin_grad_E"]:
            for langevin_lambda in L_langevin_lambda:


                output_dir = "/data/lisatmp2/alaingui/dae/generated_samples/%s/%0.6d" % (experiment_name, output_dir_counter)
                output_dir_counter += 1

                params = (script_path, pickled_dae_dir, n_samples, thinning_factor, n_E_approx_path, langevin_lambda, want_overview_plots, mcmc_method, output_dir)
                command = """python %s --pickled_dae_dir=%s --n_samples=%d --thinning_factor=%d --n_E_approx_path=%d --langevin_lambda=%f --want_overview_plots=%s --mcmc_method=%s --output_dir=%s""" % params

                print command
                r_server.rpush("train_gaussian_mixture", command)



        if mcmc_method in ["MH_svd_grad_E"]:
            for proposal_stddev in L_proposal_stddev:

                output_dir = "/data/lisatmp2/alaingui/dae/generated_samples/%s/%0.6d" % (experiment_name, output_dir_counter)
                output_dir_counter += 1

                params = (script_path, pickled_dae_dir, n_samples, thinning_factor, n_E_approx_path, proposal_stddev, want_overview_plots, mcmc_method, output_dir)
                command = """python %s --pickled_dae_dir=%s --n_samples=%d --thinning_factor=%d --n_E_approx_path=%d --proposal_stddev=%f --want_overview_plots=%s --mcmc_method=%s --output_dir=%s""" % params

                print command
                r_server.rpush("train_gaussian_mixture", command)




