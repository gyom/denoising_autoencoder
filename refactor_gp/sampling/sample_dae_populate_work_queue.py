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
#pickled_dae_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d10/experiment_07/000021"

L_pickled_dae_dir = ["/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d10/experiment_13/%0.6d" % i for i in [18,45,6,23,29,10,3]]
#L_pickled_dae_dir = ["/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d10/experiment_13/%0.6d" % i for i in [18,23,29]]
#L_pickled_dae_dir = ["/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d2/experiment_11/%0.6d" % i for i in [0, 17, 90, 5, 18]]

n_samples=500
thinning_factor=10
L_n_E_approx_path = [0]
#L_mcmc_method = ["MH_grad_E", "langevin_grad_E", "MH_langevin_grad_E", "MH_svd_grad_E"]
#L_mcmc_method = ["MH_grad_E", "MH_langevin_grad_E"]
#L_mcmc_method = ["MH_langevin_grad_E", "MH_svd_grad_E"]
L_mcmc_method = ["langevin_grad_E"]

L_langevin_stddev = [0.1]
L_langevin_beta = [1.0, 0.3, 0.1]
L_proposal_stddev = [0.1]

want_overview_plots = "True"


output_dir_counter = 0
#experiment_name = "experiment_13_02_MH_langevin_grad_E"
#experiment_name = "experiment_13_01_MH_grad_E"
experiment_name = "experiment_18_langevin"

for mcmc_method in L_mcmc_method:
    for pickled_dae_dir in L_pickled_dae_dir:

        if mcmc_method in ["langevin_grad_E", "MH_langevin_grad_E", "MH_svd_grad_E"]:
            for langevin_stddev in L_langevin_stddev:
                for langevin_beta in L_langevin_beta:
                    for n_E_approx_path in L_n_E_approx_path:
                        output_dir = "/data/lisatmp2/alaingui/dae/generated_samples/%s/%0.6d" % (experiment_name, output_dir_counter)
                        output_dir_counter += 1

                        params = (script_path, pickled_dae_dir, n_samples, thinning_factor, n_E_approx_path, langevin_stddev, langevin_beta, want_overview_plots, mcmc_method, output_dir)
                        command = """python %s --pickled_dae_dir=%s --n_samples=%d --thinning_factor=%d --n_E_approx_path=%d --langevin_stddev=%f --langevin_beta=%f --want_overview_plots=%s --mcmc_method=%s --output_dir=%s""" % params

                        print command
                        r_server.rpush("train_gaussian_mixture", command)

        if mcmc_method in ["MH_grad_E"]:
            for proposal_stddev in L_proposal_stddev:
                for n_E_approx_path in L_n_E_approx_path:
                    output_dir = "/data/lisatmp2/alaingui/dae/generated_samples/%s/%0.6d" % (experiment_name, output_dir_counter)
                    output_dir_counter += 1

                    params = (script_path, pickled_dae_dir, n_samples, thinning_factor, n_E_approx_path, proposal_stddev, want_overview_plots, mcmc_method, output_dir)
                    command = """python %s --pickled_dae_dir=%s --n_samples=%d --thinning_factor=%d --n_E_approx_path=%d --proposal_stddev=%f --want_overview_plots=%s --mcmc_method=%s --output_dir=%s""" % params

                    print command
                    r_server.rpush("train_gaussian_mixture", command)




