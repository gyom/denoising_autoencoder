#!/bin/env python

import numpy as np
import os, sys, time

import matplotlib
import pylab
import matplotlib.pyplot as plt


import refactor_gp
import refactor_gp.models
from   refactor_gp.models import dae_untied_weights

import refactor_gp.datasets
from   refactor_gp.datasets import gaussian_mixture_tools

import refactor_gp.sampling
from   refactor_gp.sampling import dispatcher


def usage():
    print ""

def main(argv):
    """

    """

    import getopt
    import cPickle
    import json

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["langevin_stddev=", "langevin_beta=", "temperature=", "output_dir=", "n_E_approx_path=", "proposal_noise_scheme=", "want_overview_plots=", "burn_in=", "thinning_factor=", "n_samples="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    thinning_factor = None
    burn_in = None
    n_samples = None
    output_dir = None
    n_E_approx_path = None
    proposal_noise_scheme = None
    langevin_stddev = None
    langevin_beta = None
    temperature = None
    want_overview_plots = False

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--mcmc_method"):
            mcmc_method = a
        elif o in ("--n_samples"):
            n_samples = int(a)
        elif o in ("--thinning_factor"):
            thinning_factor = int(a)
        elif o in ("--burn_in"):
            burn_in = int(a)
        elif o in ("--langevin_stddev"):
            langevin_stddev = float(a)
        elif o in ("--langevin_beta"):
            langevin_beta = float(a)
        elif o in ("--temperature"):
            temperature = float(a)
        elif o in ("--n_E_approx_path"):
            n_E_approx_path = float(a)
        elif o in ("--proposal_noise_scheme"):
            proposal_noise_scheme = a
        elif o in ("--want_overview_plots"):
            want_overview_plots = ((a == "True") or (a == "true") or (a == "1"))
        elif o in ("--output_dir"):
            output_dir = a
        else:
            assert False, "unhandled option"

    sampling_options = generate_sampling_options(0.1, 10)
    print sampling_options['E'](np.linspace(-5,5,20))

    sampling_options['mcmc_method'] = mcmc_method
    sampling_options['n_samples'] = n_samples

    if burn_in is not None:
        sampling_options['burn_in'] = burn_in
    if thinning_factor is not None:
        sampling_options['thinning_factor'] = thinning_factor
    #if proposal_stddev is not None:
    #    sampling_options['proposal_stddev'] = proposal_stddev
    if n_E_approx_path is not None:
        sampling_options['n_E_approx_path'] = n_E_approx_path
    if proposal_noise_scheme is not None:
        sampling_options['proposal_noise_scheme'] = proposal_noise_scheme
    #if omit_asymmetric_proposal_factor is not None:
    #    sampling_options['omit_asymmetric_proposal_factor'] = omit_asymmetric_proposal_factor
    if langevin_stddev is not None:
        sampling_options['langevin_stddev'] = langevin_stddev
    if langevin_beta is not None:
        sampling_options['langevin_beta'] = langevin_beta
    if temperature is not None:
        sampling_options['temperature'] = temperature

    print sampling_options


    res = dispatcher.mcmc_generate_samples(sampling_options)


    pylab.hold(True)
    #print samples[:,i].shape
    #print samples[:,j].shape
    pylab.scatter(samples[:,i], samples[:,j])
    pylab.draw()
    pylab.savefig(output_image_file, dpi=150)
    pylab.close()


    # list to set the kernel widths
    # plotting the samples
    # creating output dir + samples
    # writing out config
    # maybe setting seeds to be minimally diverse
    # implement KL thingy




def generate_sampling_options(train_stddev, n_components):

    # h_a(x) = 0.25 * np.exp(a * np.log(x)) * (np.tanh(A[0]*x**2 + A[1]*x + A[2]) * (np.tanh(A[3]*x**2 + A[4]*x + A[5])
    # h_b(x) = 0.25 * np.exp(b * np.log(x)) * (np.tanh(B[0]*x**2 + B[1]*x + B[2]) * (np.tanh(B[3]*x**2 + B[4]*x + B[5])
    #
    # a in [1,2]
    # b in [1,2]
    # A vector of 6 values in [-5,5]
    # B vector of 6 values in [-5,5]
    #
    # E(x) = 0.5 * x **2 + h_a(x) + h_b(x)

    def make_h():

        def h(x):
            return 0.25 * np.exp(h.A[6] * np.log(np.abs(x)+1e-8)) * np.tanh(h.A[0]*x**2 + h.A[1]*x + h.A[2]) * np.tanh(h.A[3]*x**2 + h.A[4]*x + h.A[5])

        h.A = np.zeros((7,))
        for i in range(6):
            h.A[i] = np.random.uniform(low=-5.0, high=5.0)
        h.A[6] = np.random.uniform(low=1.0, high=2.0)

        # it's not important to have the exact value here because
        # we're not dealing with precisions less than 1e-8
        def h_prime(x):
            return (h(x+1e-8) - h(x))/1e-8

        return {'h':h, 'h_prime':h_prime}

    def make_combination_h(N):
        make_combination_h.L = [make_h() for n in range(N)]
        def h(x):
            return reduce(lambda a, b: a+b, [e['h'](x) for e in make_combination_h.L])
        def h_prime(x):
            return reduce(lambda a, b: a+b, [e['h_prime'](x) for e in make_combination_h.L])
        return {'h':h, 'h_prime':h_prime}

    def E(x):
        return 0.5*x**2 - E.h(x)

    def grad_E(x):
        return x - grad_E.h_prime(x)

    D = make_combination_h(n_components)
    grad_E.h_prime = D['h_prime']
    E.h = D['h']

    def r(x):
        return x - train_stddev**2 * grad_E(x)

    sampling_options = { 'r':r,
                         'E':E,
                         'grad_E':grad_E,
                         'train_stddev':train_stddev}

    return sampling_options







# bla bla





if __name__ == "__main__":
    main(sys.argv)
