
import numpy as np
import time
import scipy

import refactor_gp
import refactor_gp.sampling
import refactor_gp.sampling.metropolis_hastings.langevin as langevin
import refactor_gp.sampling.metropolis_hastings.vanilla as vanilla
import refactor_gp.sampling.metropolis_hastings.svd as svd

import refactor_gp.gyom_utils
from refactor_gp.gyom_utils import get_dict_key_or_default


def mcmc_generate_samples(sampling_options):

    train_stddev    = get_dict_key_or_default(sampling_options, 'train_stddev', None, True)
    proposal_stddev = get_dict_key_or_default(sampling_options, 'proposal_stddev', None)
    n_samples       = get_dict_key_or_default(sampling_options, 'n_samples',       None, True)
    thinning_factor = get_dict_key_or_default(sampling_options, 'thinning_factor', 100)
    burn_in         = get_dict_key_or_default(sampling_options, 'burn_in',         n_samples * thinning_factor / 10)
    proposal_noise_scheme = get_dict_key_or_default(sampling_options, 'proposal_noise_scheme', None)
    omit_asymmetric_proposal_factor = get_dict_key_or_default(sampling_options, 'omit_asymmetric_proposal_factor', None)

    langevin_stddev = get_dict_key_or_default(sampling_options, 'langevin_stddev', None)
    langevin_beta   = get_dict_key_or_default(sampling_options, 'langevin_beta',   None)
    temperature     = get_dict_key_or_default(sampling_options, 'temperature',     None)
    mcmc_method     = get_dict_key_or_default(sampling_options, 'mcmc_method',     None, True)
    x0              = get_dict_key_or_default(sampling_options, 'x0',              np.random.normal(size=(2,)))
    n_chains        = get_dict_key_or_default(sampling_options, 'n_chains',        None)

    # E would be something like ninja_start_distribution.E, and grad_E would be ninja_start_distribution.grad_E
    E               = get_dict_key_or_default(sampling_options, 'E',               None)
    grad_E          = get_dict_key_or_default(sampling_options, 'grad_E',          None)
    n_E_approx_path = get_dict_key_or_default(sampling_options, 'n_E_approx_path', None)

    # applicable only when dealing with a DAE
    f               = get_dict_key_or_default(sampling_options, 'f',               None)
    f_prime         = get_dict_key_or_default(sampling_options, 'f_prime',         None)

    # use when given
    r               = get_dict_key_or_default(sampling_options, 'r',               None)
    r_prime         = get_dict_key_or_default(sampling_options, 'r_prime',         None)


    # In the process of cleaning up that mess of having two different names
    # for basically the same thing.
    #if (proposal_stddev == None) and not (langevin_stddev == None):
    #    proposal_stddev = langevin_stddev
    #elif (langevin_stddev == None) and not (proposal_stddev == None):
    #    langevin_stddev = proposal_stddev

    # Run a sanity check to be sure that E and grad_E take the correct input dimension given by x0.
    # This could later be used to poke around to see if E and grad_E are vectorial or not.

    # We accept values for x0 that are of size (n_chains, d) or (d,).

    d = x0.shape[-1]
    # If we asked for more than one chain, but x0 contains
    # only one initial state, when we will expand it to be
    # of shape (n_chains, d)
    #
    # If n_chains is None, we'll leave x0 alone and we'll
    # return samples of size (n_samples, d) instead of
    # (n_chains, n_samples, d).

    if n_chains == None:
        n_chains = 1
        want_vectorial_result = False
    else:
        want_vectorial_result = True

    if len(x0.shape) == 1:
        x0 = np.tile(x0.reshape((1,d)),(n_chains,1))

    # Some of those methods might not be defined properly
    # in situations where we don't have the energy known.
    # However, they won't be used in such cases.
    def approximate_energy_difference(proposed_x, current_x):
        return grad_E(current_x).dot(proposed_x - current_x)
    #
    def exact_energy_difference(proposed_x, current_x):
        return E(proposed_x) - E(current_x)
    #
    def zero_energy_difference(proposed_x, current_x):
        return 0.0
    def approximate_energy_difference_by_path_integration(proposed_x, current_x):
        # We approximate the energy difference by integrating the gradient
        # over a path from current_x to proposed_x. The path used is
        # a simple line segment. Original idea suggested by Yutian.
        # The number of points used for the approximation of the path
        # is baked into this function.
        d = proposed_x.shape[0]
        t = np.tile(np.linspace(0,1,n_E_approx_path).reshape((-1,1)),
                    (1,d))
        path = (1 - t) * current_x.reshape((1,-1)) + t * proposed_x.reshape((1,-1))
        # could be turned into something better if we had a vectorial
        # implementation of grad_E
        grad_E_path = np.vstack( [grad_E(v) for v in path] ).mean(axis=0)
        assert grad_E_path.shape == (d,)
        # debug
        #print "%f, %f" % (grad_E_path.dot(proposed_x - current_x).mean(),
        #                  approximate_energy_difference(proposed_x, current_x) )

        return grad_E_path.dot(proposed_x - current_x)


    samples_for_all_chains = np.zeros((n_chains, n_samples, d))
    acceptance_ratio_list = []
    sampling_start_time = time.time()

    for c in np.arange(n_chains):

        if mcmc_method == 'MH_grad_E':
            assert proposal_stddev > 0

            noise_levels = {'train_stddev':train_stddev,
                            'proposal_stddev':proposal_stddev}
            if not temperature == None:
                noise_levels["temperature"] = temperature

            if n_E_approx_path and n_E_approx_path > 1:
                energy_difference = approximate_energy_difference_by_path_integration
            else:
                energy_difference = approximate_energy_difference

            (X, acceptance_ratio) = vanilla.sample_chain(x0[c,:], n_samples, energy_difference, proposal_stddev, thinning_factor = thinning_factor, burn_in = burn_in, temperature = temperature)
        elif mcmc_method == 'langevin_grad_E':
            assert r
            assert r_prime
            noise_levels = {'train_stddev':train_stddev}
            if not langevin_stddev == None:
                noise_levels["langevin_stddev"] = langevin_stddev
            if not langevin_beta == None:
                noise_levels["langevin_beta"] = langevin_beta
            if not temperature == None:
                noise_levels["temperature"] = temperature
            noise_levels = fill_missing_values_in_langevin_noise_levels(noise_levels)

            if n_E_approx_path and n_E_approx_path > 1:
                energy_difference = approximate_energy_difference_by_path_integration
            else:
                energy_difference = approximate_energy_difference

            (X, acceptance_ratio, noise_levels) = langevin.sample_chain(x0[c,:], n_samples, energy_difference, noise_levels, r, r_prime, thinning_factor = thinning_factor, burn_in = burn_in, accept_all_proposals = True, proposal_noise_scheme = proposal_noise_scheme, omit_asymmetric_proposal_factor=omit_asymmetric_proposal_factor)

        elif mcmc_method == 'MH_langevin_grad_E':
            assert r
            assert r_prime
            noise_levels = {'train_stddev':train_stddev}
            if not langevin_stddev == None:
                noise_levels["langevin_stddev"] = langevin_stddev
            if not langevin_beta == None:
                noise_levels["langevin_beta"] = langevin_beta
            if not temperature == None:
                noise_levels["temperature"] = temperature
            noise_levels = fill_missing_values_in_langevin_noise_levels(noise_levels)

            if n_E_approx_path and n_E_approx_path > 1:
                energy_difference = approximate_energy_difference_by_path_integration
            else:
                energy_difference = approximate_energy_difference

            (X, acceptance_ratio, noise_levels) = langevin.sample_chain(x0[c,:], n_samples, energy_difference, noise_levels, r, r_prime, thinning_factor = thinning_factor, burn_in = burn_in, proposal_noise_scheme=proposal_noise_scheme, omit_asymmetric_proposal_factor=omit_asymmetric_proposal_factor)

        elif mcmc_method == "MH_svd_grad_E":
            assert f_prime
            assert r_prime
            assert r
            noise_levels = {'train_stddev':train_stddev}
            if not langevin_stddev == None:
                noise_levels["langevin_stddev"] = langevin_stddev
            if not langevin_beta == None:
                noise_levels["langevin_beta"] = langevin_beta
            if not temperature == None:
                noise_levels["temperature"] = temperature
            noise_levels = fill_missing_values_in_langevin_noise_levels(noise_levels)

            if n_E_approx_path and n_E_approx_path > 1:
                energy_difference = approximate_energy_difference_by_path_integration
            else:
                energy_difference = approximate_energy_difference

            (X, acceptance_ratio, noise_levels) = svd.sample_chain(x0[c,:], n_samples, energy_difference, noise_levels, r, r_prime, f_prime, thinning_factor = thinning_factor, burn_in = burn_in, proposal_noise_scheme=proposal_noise_scheme, omit_asymmetric_proposal_factor=omit_asymmetric_proposal_factor)
        else:
            raise("Unrecognized value for parameter 'mcmc_method' : %s" % (mcmc_method,))

        samples_for_all_chains[c,:,:] = X
        acceptance_ratio_list.append(acceptance_ratio)
    # end of for loop

    sampling_end_time = time.time()

    combined_acceptance_ratio = np.mean(np.array(acceptance_ratio_list))
    #print "Got the samples. Acceptance ratio was %f" % combined_acceptance_ratio
    proposals_per_second = (n_chains * n_samples * thinning_factor + burn_in) / (sampling_end_time - sampling_start_time)
    #print "MCMC proposal speed was 10^%0.2f / s" % (np.log(proposals_per_second) / np.log(10), )

    if not want_vectorial_result:
        assert samples_for_all_chains.shape[0] == 1
        samples_for_all_chains = samples_for_all_chains[0,:,:]

    return {'samples': samples_for_all_chains,
            'elapsed_time':sampling_end_time - sampling_start_time,
            'proposals_per_second':proposals_per_second,
            'acceptance_ratio':combined_acceptance_ratio,
            'noise_levels':noise_levels}


def fill_missing_values_in_langevin_noise_levels(noise_levels, want_print_values=True):
    """
    train_stddev
    langevin_stddev
    langevin_beta
    temperature
    """

    def print_values():
        if want_print_values:
            print "==========================="
            print "With your current setup, you have that the sampling procedure scales as follows."
            print ""
            print "train_stddev : %f" % train_stddev
            print "langevin_stddev : %f" % langevin_stddev
            print 
            print "langevin_beta : %f" % langevin_beta
            print "temperature : %f" % temperature
            print "==========================="

    fields = ['train_stddev', 'langevin_stddev', 'langevin_beta', 'temperature']

    assert noise_levels.has_key("train_stddev")
    train_stddev = noise_levels["train_stddev"]

    # make sure it doesn't have anything funny included in it
    for key in noise_levels.keys():
        assert key in fields

    langevin_stddev = get_dict_key_or_default(noise_levels, 'langevin_stddev', None)
    langevin_beta = get_dict_key_or_default(noise_levels, 'langevin_beta', None)
    temperature = get_dict_key_or_default(noise_levels, 'temperature', None)

    nbr_keys = len(noise_levels.keys())
    if nbr_keys == 4:
        # nothing to do except maybe validating them ?

        print_values()
        return {'train_stddev':train_stddev,
                'langevin_stddev':langevin_stddev,
                'langevin_beta':langevin_beta,
                'temperature':temperature}

    if nbr_keys == 3:
        # find out which one is missing and solve for that one
        if langevin_stddev is None:
            langevin_stddev = np.sqrt(2*temperature*langevin_beta)*train_stddev
        elif langevin_beta is None:
            langevin_beta = langevin_stddev**2 / (2*temperature*train_stddev**2)
        elif temperature is None:
            temperature = langevin_stddev**2 / ( 2 * langevin_beta * train_stddev**2 )
        else:
            assert False
        
        print_values()
        return {'train_stddev':train_stddev,
                'langevin_stddev':langevin_stddev,
                'langevin_beta':langevin_beta,
                'temperature':temperature}

    elif nbr_keys <= 2:
        # Fill out one value first and call this function
        # recursively to compute the other missing field.
        #
        # We decide to prioritize setting the temperature to 1
        # instead of the somewhat meaningless quantity beta.
        if temperature is None:
            temperature = 1.0
        elif langevin_beta is None:
            langevin_beta = 1.0
        else:
            assert False

        return fill_missing_values_in_langevin_noise_levels( {'train_stddev':train_stddev,
                                                              'langevin_stddev':langevin_stddev,
                                                              'langevin_beta':langevin_beta,
                                                              'temperature':temperature} )