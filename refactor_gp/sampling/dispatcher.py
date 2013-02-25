
import numpy as np
import time
import scipy

from gyom_utils import get_dict_key_or_default

import refactor_gp
import refactor_gp.sampling
import refactor_gp.sampling.metropolis_hastings.langevin as langevin
import refactor_gp.sampling.metropolis_hastings.vanilla as vanilla
import refactor_gp.sampling.metropolis_hastings.svd as svd


"""
if not E == None:
    # This would be the regular branch executed when
    # using Monte Carlo.
    energy_difference = exact_energy_difference
elif E == None and not grad_E == None:
    energy_difference = approximate_energy_difference
else:
    raise("Unrecognized setup.")
"""




def mcmc_generate_samples(sampling_options):

    proposal_stddev = get_dict_key_or_default(sampling_options, 'proposal_stddev', None)
    n_samples       = get_dict_key_or_default(sampling_options, 'n_samples',       None, True)
    thinning_factor = get_dict_key_or_default(sampling_options, 'thinning_factor', 100)
    burn_in         = get_dict_key_or_default(sampling_options, 'burn_in',         n_samples * thinning_factor / 10)
    langevin_lambda = get_dict_key_or_default(sampling_options, 'langevin_lambda', None)
    mcmc_method     = get_dict_key_or_default(sampling_options, 'mcmc_method',     None, True)
    x0              = get_dict_key_or_default(sampling_options, 'x0',              np.random.normal(size=(2,)))
    n_chains        = get_dict_key_or_default(sampling_options, 'n_chains',        None)

    # E would be something like ninja_start_distribution.E, and grad_E would be ninja_start_distribution.grad_E
    E               = get_dict_key_or_default(sampling_options, 'E',               None)
    grad_E          = get_dict_key_or_default(sampling_options, 'grad_E',          None)

    # applicable only when dealing with a DAE
    f               = get_dict_key_or_default(sampling_options, 'f',               None)
    f_prime         = get_dict_key_or_default(sampling_options, 'f_prime',         None)

    # use when given
    r               = get_dict_key_or_default(sampling_options, 'r',               None)
    r_prime         = get_dict_key_or_default(sampling_options, 'r_prime',         None)


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
    # def approximate_energy_difference_by_integration(proposed_x, current_x):
    #    do something like a linspace summation with approximate_energy_difference
    #    as suggested by Yutian
    
    samples_for_all_chains = np.zeros((n_chains, n_samples, d))
    acceptance_ratio_list = []
    sampling_start_time = time.time()

    for c in np.arange(n_chains):

        #if mcmc_method == 'metropolis_hastings_E':
        #    assert proposal_stddev > 0.0
        #    symmetric_proposal = lambda x: x + np.random.normal(size=x.shape, scale = proposal_stddev)
        #    (X, acceptance_ratio) = run_chain_with_energy(E, x0[c,:], symmetric_proposal, n_samples, thinning_factor = thinning_factor, burn_in = burn_in)
        #
        #elif mcmc_method == 'metropolis_hastings_grad_E':
        #    assert proposal_stddev > 0.0
        #    symmetric_proposal = lambda x: x + np.random.normal(size=x.shape, scale = proposal_stddev)
        #    (X, acceptance_ratio) = run_chain_with_energy(None, x0[c,:], symmetric_proposal, n_samples, thinning_factor = thinning_factor, burn_in = burn_in, grad_E = grad_E)
        #
        if mcmc_method == 'langevin_grad_E':
            assert r
            assert langevin_lambda > 0
            (X, acceptance_ratio) = langevin.sample_chain(x0[c,:], n_samples, approximate_energy_difference, langevin_lambda, r, thinning_factor = thinning_factor, burn_in = burn_in, accept_all_proposals = True)
        #
        #elif mcmc_method == 'MH_langevin_E':
        #
        #    # grad_E is needed to define the perfect reconstruction function
        #    # E is used for the Metropolis-Hastings sampling
        #
        #    # remember that we have a minus here because it's d log p(x) / dx
        #    # which corresponds to - dE(x) / dx
        #    r = lambda x: x - langevin_lambda * grad_E(x)
        #   (X, acceptance_ratio) = run_chain_with_langevin_proposals(x0[c,:], n_samples, langevin_lambda, E = E, thinning_factor = thinning_factor, burn_in = burn_in, r = r)
        #
        #    #(X, acceptance_ratio) = metropolis_hastings_sampler.run_chain_with_energy(ninja_star_distribution.E, x0, None, n_samples, thinning_factor = thinning_factor, burn_in = burn_in, asymmetric_proposal = asymmetric_proposal)
        #
        elif mcmc_method == 'MH_langevin_grad_E':
            assert r
            assert langevin_lambda > 0
            (X, acceptance_ratio) = langevin.sample_chain(x0[c,:], n_samples, approximate_energy_difference, langevin_lambda, r, thinning_factor = thinning_factor, burn_in = burn_in)

        elif mcmc_method == "MH_svd_grad_E":
            assert f_prime
            assert r_prime
            assert r
            assert proposal_stddev > 0
            (X, acceptance_ratio) = svd.sample_chain(x0[c,:], n_samples, approximate_energy_difference, proposal_stddev = proposal_stddev, r = r, r_prime = r_prime, f_prime = f_prime, thinning_factor = thinning_factor, burn_in = burn_in)
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
            'acceptance_ratio':combined_acceptance_ratio    }


    