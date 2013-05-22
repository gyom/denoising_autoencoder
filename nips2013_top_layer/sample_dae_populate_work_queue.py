#!/bin/env python

import redis
import numpy as np

r_server = redis.Redis("localhost", 6379)

if not r_server.ping():
    print "Cannot ping server. Exiting."
    quit()


script_path = "/u/alaingui/umontreal/denoising_autoencoder/refactor_gp/sampling/sample_dae.py"

#################
exact_grad_E_from_mixture_mvn_pickle = None
#exact_grad_E_from_mixture_mvn_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d10_eig0.1_comp25_001/train_samples_extra.pkl"
#exact_grad_E_from_mixture_mvn_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d2_eig0.1_comp25_001/train_samples_extra.pkl"

L_pickled_dae_dir = ["/data/lisatmp2/alaingui/dae/dae_trained_models/mnist_yann_H1/experiment_03_yann_mnist_H1/%0.6d" % i for i in [23, 19, 22, 17, 18, 16]]
#L_pickled_dae_dir = ["/data/lisatmp2/alaingui/dae/dae_trained_models/mnist_yann_H1/experiment_03_yann_mnist_H1/%0.6d" % i for i in [0, 4, 1, 3, 6, 5, 2, 7]]
##################

#experiment_name = "experiment_34_MH_langevin"
experiment_name = "experiment_03_yann_mnist_H1_02"
def output_dir_generator(i):
    return "/data/lisatmp2/alaingui/dae/generated_samples/%s/%0.6d" % (experiment_name, i)

##################

n_samples=5000
thinning_factor=10
L_n_E_approx_path = [20]

#L_mcmc_method = ["langevin_grad_E"]
L_mcmc_method = ["MH_langevin_grad_E"]
#L_mcmc_method = ["MH_svd_grad_E"]

L_langevin_stddev = [0.1, 0.01]
L_fake_train_stddev = [0.1, 0.01]
#L_langevin_beta = [0.1, 0.05]
L_langevin_beta = []
L_omit_asymmetric_proposal_factor = [0,1]
#L_proposal_noise_scheme = ['merge_x', 'noise_E', 'noise_r']
L_proposal_noise_scheme = ['merge_x']
L_temperature = [5, 1, 0.2]

L_proposal_stddev = [0.1, 0.01]

want_overview_plots = "False"

##################

def c_all():
    return c_output_dir(c_method(c_unconditionals(c_model(c_root()))))

def c_root():
    return ["python " + script_path]

def c_model(L):
    return (  [e + " --exact_grad_E_from_mixture_mvn_pickle=%s --fake_train_stddev=%f " % (exact_grad_E_from_mixture_mvn_pickle, fake_train_stddev)
               for e in L for fake_train_stddev in L_fake_train_stddev if (exact_grad_E_from_mixture_mvn_pickle is not None)] +
              [e + " --pickled_dae_dir=%s " % (pickled_dae_dir,)
               for e in L for pickled_dae_dir in L_pickled_dae_dir if (L_pickled_dae_dir is not None)] )

def c_unconditionals(L):
    return [e + " --n_samples=%d --n_E_approx_path=%d --thinning_factor=%d --want_overview_plots=%s " % (n_samples, n_E_approx_path, thinning_factor, want_overview_plots)
            for e in L for n_E_approx_path in L_n_E_approx_path]

def c_output_dir(L):
    return [e + " --output_dir=%s" % (output_dir_generator(i),) for (e,i) in zip(L, range(len(L)))]

def c_method(L):

    res = []
    for mcmc_method in L_mcmc_method:
        if mcmc_method in ["MH_grad_E"]:
            res = res + c_temperature(c_proposal_stddev([e + " --mcmc_method=%s " % (mcmc_method, ) for e in L]))
        elif mcmc_method in ["langevin_grad_E"]:
            res = res + c_proposal_noise_scheme(c_temperature(c_langevin_stddev(c_langevin_beta([e + " --mcmc_method=%s " % (mcmc_method, ) for e in L]))))
        elif mcmc_method in ["MH_langevin_grad_E", "MH_svd_grad_E"]:
            res = res + c_omit_asymmetric_proposal_factor(c_proposal_noise_scheme(c_temperature(c_langevin_stddev(c_langevin_beta([e + " --mcmc_method=%s " % (mcmc_method, ) for e in L])))))
    return res

def c_proposal_stddev(L):
    if len(L_proposal_stddev) > 0:
        return [e + " --proposal_stddev=%f" % (proposal_stddev,) for e in L for proposal_stddev in L_proposal_stddev]
    else:
        return L

def c_temperature(L):
    if len(L_temperature) > 0:
        return [e + " --temperature=%f" % (temperature,) for e in L for temperature in L_temperature]
    else:
        return L

def c_langevin_stddev(L):
    if len(L_langevin_stddev) > 0:
        return [e + " --langevin_stddev=%f" % (langevin_stddev,) for e in L for langevin_stddev in L_langevin_stddev]
    else:
        return L

def c_langevin_beta(L):
    if len(L_langevin_beta) > 0:
        return [e + " --langevin_beta=%f" % (langevin_beta,) for e in L for langevin_beta in L_langevin_beta]
    else:
        return L

def c_proposal_noise_scheme(L):
    if len(L_proposal_noise_scheme) > 0:
        return [e + " --proposal_noise_scheme=%s" % (proposal_noise_scheme,) for e in L for proposal_noise_scheme in L_proposal_noise_scheme]
    else:
        return L

def c_proposal_noise_scheme(L):
    if len(L_proposal_noise_scheme) > 0:
        return [e + " --proposal_noise_scheme=%s" % (proposal_noise_scheme,) for e in L for proposal_noise_scheme in L_proposal_noise_scheme]
    else:
        return L

def c_omit_asymmetric_proposal_factor(L):
    if len(L_omit_asymmetric_proposal_factor) > 0:
        return [e + " --omit_asymmetric_proposal_factor=%d" % (omit_asymmetric_proposal_factor,) for e in L for omit_asymmetric_proposal_factor in L_omit_asymmetric_proposal_factor]
    else:
        return L


for cmd in c_all():
    print cmd
    r_server.rpush("train_yann_mnist", cmd)



