#!/bin/env python

import numpy as np
import os, sys, time

import refactor_gp
import refactor_gp.models
from   refactor_gp.models import dae_untied_weights

import refactor_gp
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
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["pickled_dae_dir=", "mcmc_method=", "n_samples=", "thinning_factor=", "proposal_stddev=", "output_dir="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    pickled_dae_dir = None
    mcmc_method = None
    n_samples = None
    thinning_factor = None
    burn_in = None
    proposal_stddev = None
    output_dir = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--pickled_dae_dir"):
            pickled_dae_dir = a
        elif o in ("--mcmc_method"):
            mcmc_method = a
        elif o in ("--n_samples"):
            n_samples = int(a)
        elif o in ("--thinning_factor"):
            thinning_factor = int(a)
        elif o in ("--burn_in"):
            burn_in = int(a)
        elif o in ("--proposal_stddev"):
            proposal_stddev = float(a)
        elif o in ("--output_dir"):
            output_dir = a
        else:
            assert False, "unhandled option"
 
    #print "want_early_termination is %d" % want_early_termination

    assert os.path.exists(pickled_dae_dir)
    assert os.path.isdir(pickled_dae_dir)

    pickled_dae_file = os.path.join(pickled_dae_dir, "trained_dae.pkl")
    pickled_dae_extra_file = os.path.join(pickled_dae_dir, "extra_details.pkl")
    assert os.path.exists(pickled_dae_file)
    assert os.path.exists(pickled_dae_extra_file)

    # irrelevant values because we load a pickle anyways
    mydae = dae_untied_weights.DAE_untied_weights(n_inputs = 1,
                                                  n_hiddens = 1,
                                                  act_func = ['tanh', 'tanh'])
    mydae.load_pickle(pickled_dae_file)
    extra_details = cPickle.load(open(pickled_dae_extra_file, "r"))

    # Here's an example of what extra_details contains.
    """
    {'act_func': [u'tanh', u'id'],
 'computational_cost_in_seconds': 373,
 'early_termination_occurred': True,
 'lbfgs_rank': 4,
 'maxiter': 1000,
 'model_losses': [0.88493400434004688,  0.085861144685915214,  0.014545763882193379,  0.0033357369919251481,  0.0003573931475000403,  3.3169528288705622e-05,  nan,  nan,  nan,  nan],
 'n_hiddens': 30,
 'n_inputs': 10,
 'noise_stddevs': [1.0,  0.2782559402207124,  0.07742636826811268,  0.02154434690031883,  0.005994842503189405,  0.0016681005372000581,  0.00046415888336127757,  0.00012915496650148825,  3.5938136638046215e-05,  9.99999999999998e-06],
 'train_samples_pickle': '/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d10_eig0.1_comp25_001/train_samples.pkl'}
    """

    n_inputs = extra_details['n_inputs']
    n_hiddens = extra_details['n_hiddens']

    langevin_lambda = None
    for (loss, noise_stddev) in zip(extra_details['model_losses'],
                                    extra_details['noise_stddevs']):
        if np.isnan(loss):
            break
        else:
            langevin_lambda = noise_stddev

    # At that point, langevin_lambda should be the associated
    # value for that DAE that we loaded. If langevin_lambda is
    # still None, it means that we've loaded a model that was
    # not trained properly.
    assert langevin_lambda
    print "Using langevin_lambda %f" % (langevin_lambda,)

    dae_params = read_parameters_from_dae(mydae)
    r = dae_params['r']

    # don't forget the minus sign here !
    # remember that r(x) - x is propto  -1 * dE/dx
    grad_E = lambda x: -1*(r(x)-x) / langevin_lambda

    # We always sample from the origin.
    x0 = np.zeros((n_inputs,))


    sampling_options = {'x0':x0,
                        'f_prime':dae_params['f_prime'],
                        'r':dae_params['r'],
                        'r_prime':dae_params['r_prime'],
                        'mcmc_method':mcmc_method,
                        'grad_E':grad_E,
                        'n_samples':n_samples,
                        'langevin_lambda':langevin_lambda}
    if burn_in:
        sampling_options['burn_in'] = burn_in
    if thinning_factor:
        sampling_options['thinning_factor'] = thinning_factor
    if proposal_stddev:
        sampling_options['proposal_stddev'] = proposal_stddev

    #### Perform the sampling. ####

    res = dispatcher.mcmc_generate_samples(sampling_options)
    #
    # res has keys : samples
    #                elapsed_time
    #                proposals_per_second
    #                acceptance_ratio

    print "Sampling took %d seconds." % (res['elapsed_time'],)
    print ""
    print "Proposals per second : 10^%0.2f." % (np.log(res['proposals_per_second'])/np.log(10),)
    print "Acceptance ratio : %0.3f." % (res['acceptance_ratio'],)


    #### Write out the results. ####

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print "Creating directory %s" % (output_dir,)

    samples_file = os.path.join(output_dir, "mcmc_samples.pkl")
    cPickle.dump(res['samples'], open(samples_file, "w"))
    print "Wrote samples in %s" % (samples_file,)

    sampling_extra_details = res
    # Get rid of the memory-intensive 'samples'
    # field because it was already written in mcmc_samples.pkl.
    del sampling_extra_details['samples']
    sampling_extra_details['pickled_dae_dir'] = pickled_dae_dir
    sampling_extra_details['mcmc_method'] = mcmc_method
    sampling_extra_details['n_samples'] = n_samples
    sampling_extra_details['thinning_factor'] = thinning_factor
    sampling_extra_details['burn_in'] = burn_in
    sampling_extra_details['proposal_stddev'] = proposal_stddev

    sampling_extra_pickle_file = os.path.join(output_dir, "sampling_extra_details.pkl")
    sampling_extra_json_file = os.path.join(output_dir, "sampling_extra_details.json")
    cPickle.dump(extra_details, open(sampling_extra_pickle_file, "w"))
    json.dump(extra_details, open(sampling_extra_json_file, "w"))
    print "Wrote %s" % (sampling_extra_pickle_file,)
    print "Wrote %s" % (sampling_extra_json_file,)


    #### Copy the original DAE. ####

    # We'll copy the original DAE here because we
    # want to make sure that it stays around when
    # we want to inspect the samples and check the
    # characteristics of the DAE.

    import shutil
    output_dir_subdir = os.path.join(output_dir, "trained_dae")
    shutil.copytree(pickled_dae_dir, output_dir_subdir)
    print "Transfered trained DAE from %s to %s" % (pickled_dae_dir, output_dir_subdir)

    print "Done."





def read_parameters_from_dae(some_dae):
    """
    This be a tool to get the functions r, r_prime, f_prime
    from a dae like DAE_untied_weights.
    """
    def r(x):
        # only asserted because that's what we expect,
        # not asserted because it would produce some conceptual
        # problem
        assert len(x.shape) == 1
        return some_dae.encode_decode(x.reshape((1,-1))).reshape((-1,))
    
    def r_prime(x):
        return some_dae.jacobian_encode_decode(x)
    
    def f(x):
        assert len(x.shape) == 1
        return some_dae.encode(x.reshape((1,-1))).reshape((-1,))
            
    def f_prime(x):
        return some_dae.jacobian_encode(x)

    return {'r':r, 'r_prime':r_prime,
            'f':f, 'f_prime':f_prime}


if __name__ == "__main__":
    main(sys.argv)
