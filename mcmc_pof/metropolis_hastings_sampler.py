
import numpy as np
import time
import scipy

def run_chain_with_energy(E, x0, symmetric_proposal, N, thinning_factor = 1, burn_in = 0, grad_E = None, asymmetric_proposal = None):
    """
    Will sample N values for the chain starting with x0.
    The energy function is given by E(...) which takes a
    vector similar to x0 as argument.

    'symmetric_proposal' is a function that yields the
    next position proposed. It is assumed to be symmetrical
    because we don't compensate in the ratio.

    'grad_E' is the gradient of the energy function.
    It is used as an approximate replacement when
    the argument 'E' is set to be None.

    'asymmetric_proposal' is a function that yields a pair
    that contains the next proposed state and the value of
    log q( current_x | proposed_x ) - log q( proposed_x | current_x )
    which can be included directly in the computation of the
    acceptance criterion
    """

    if len(x0.shape) != 1:
        error("Wrong dimension for x0. This function is not vectorial.")

    if thinning_factor < 1:
        error("You misunderstood the thinning_factor. It should be 1 for no thinning, and 32 if we want one out of every 32 samples.")

    def approximate_energy_difference(proposed_x, current_x):
        return grad_E(current_x).dot(proposed_x - current_x)

    def exact_energy_difference(proposed_x, current_x):
        return E(proposed_x) - E(current_x)


    if E == None:
        if grad_E == None:
            error("If you don't specify the energy function, you need to at least provide the gradient of the energy function.")
        else:
            energy_difference = approximate_energy_difference
    else:
        # This would be the regular branch executed when
        # using Monte Carlo.
        energy_difference = exact_energy_difference

    if symmetric_proposal == None:
        if asymmetric_proposal == None:
            error("You need to specify either a symmetric or an asymmetric prosal.")
        else:
            proposal = asymmetric_proposal
    else:
        proposal = lambda current_x: (symmetric_proposal(current_x), 0.0)


    def iterate_N_times(current_x, energy_difference, N):
        for _ in np.arange(N):
            (proposed_x, asymmetric_correction_log_factor) = proposal(current_x)
            #proposed_x = symmetric_proposal(current_x)
            loga = - energy_difference(proposed_x, current_x) + asymmetric_correction_log_factor
            if loga >= 0 or loga >= np.log(np.random.uniform(0,1)):
                # accepted !
                current_x = proposed_x
                iterate_N_times.accepted_counter += 1
            else:
                iterate_N_times.rejected_counter += 1

        return current_x

    iterate_N_times.accepted_counter = 0
    iterate_N_times.rejected_counter = 0


    # Start with the burn-in iterations.
    current_x = x0
    current_x = iterate_N_times(current_x, energy_difference, burn_in)

    # Then we can think about collecting samples.
    samples_list = []
    # Start from the 'current_x' from the burn_in
    # and not from x0. Reset the acceptance counters.
    iterate_N_times.accepted_counter = 0
    iterate_N_times.rejected_counter = 0

        
    for n in np.arange(0,N):
        current_x = iterate_N_times(current_x, energy_difference, thinning_factor)
        # collect sample after running through the thinning iterations
        samples_list.append(current_x)


    samples = np.vstack(samples_list)
    acceptance_ratio = iterate_N_times.accepted_counter * 1.0 / (iterate_N_times.accepted_counter + iterate_N_times.rejected_counter)

    return (samples, acceptance_ratio)


# TODO : Eliminate this function once you're convinced that
#        it's not being used.
def make_langevin_sampler_requirements(langevin_lambda, grad_E=None, r=None):
    """
    This function takes care of packaging a certain number
    of objects needed to use the metropolis_hastings_sampler
    as a langevin sampler.

    This function does not do certain things which may be useful
    to do langevin with a 100% acceptance ratio. This would
    involve passing the 'run_chain_with_energy' function an energy
    function that is constant, for example, to fool it into
    accepting everything (while at the same time running an
    asymmetric_proposal that takes into consideration the actual
    gradient of the energy function).
    """
    
    if grad_E == None and r == None:
        error("Missing either grad_E or the reconstruction function.")

    if r == None:
        # Note that the resulting function 'r' can be applied
        # to a whole collected of stacked values of x when
        # grad_E accommodates it.
        r = lambda x: x - langevin_lambda * grad_E(x)

    def asymmetric_proposal(current_x):

        r_current_x = r(current_x)

        # Refer to our paper for an examplation on the factor 2.0 in there.
        proposed_x = r_current_x + np.random.normal(size=current_x.shape, scale=np.sqrt(2*langevin_lambda))

        r_proposed_x = r(proposed_x)

        # Now we need to compute
        # log q( current_x | proposed_x ) - log q( proposed_x | current_x )

        asymmetric_correction_log_factor = 1.0/(4*langevin_lambda)*( - ((current_x - r_proposed_x)**2).sum() + ((proposed_x - r_current_x)**2).sum())

        return (proposed_x, asymmetric_correction_log_factor)

    return (asymmetric_proposal, r)




def run_chain_with_langevin_proposals(x0, N, langevin_lambda, E = None, grad_E = None, thinning_factor = 1, burn_in = 0, r = None, accept_all_proposals = False):
    """
    Will sample N values for the chain starting with x0.
    The energy function is given by E(...) which takes a
    vector similar to x0 as argument.

    'grad_E' is the gradient of the energy function.
    It is used as an approximate replacement when
    the argument 'E' is set to be None.
    """

    assert len(x0.shape) == 1, "Wrong dimension for x0."

    assert thinning_factor >= 1, "You misunderstood the thinning_factor. It should be 1 for no thinning, and 32 if we want one out of every 32 samples."

    def approximate_energy_difference(proposed_x, current_x):
        return grad_E(current_x).dot(proposed_x - current_x)

    def exact_energy_difference(proposed_x, current_x):
        return E(proposed_x) - E(current_x)

    def zero_energy_difference(proposed_x, current_x):
        return 0.0


    assert langevin_lambda > 0.0, "It doesn't make sense to have the langevin lambda be less than 0. With 0.0, a lot becomes pointless."

    if not E == None:
        # This would be the regular branch executed when
        # using Monte Carlo.
        energy_difference = exact_energy_difference
    elif E == None and not grad_E == None:
        energy_difference = approximate_energy_difference
    elif E == None and grad_E == None and not r == None:
        # This is the approximation that plays the most important
        # role in our paper.
        grad_E = lambda x: - (r(x) - x) / langevin_lambda
        energy_difference = zero_energy_difference
    elif E == None and grad_E == None and r == None:
        raise("You need to provide E, grad_E or r.")
    else:
        raise("Unrecognized setup.")

    if r == None:
        # Note that the resulting function 'r' can be applied
        # to a whole collected of stacked values of x when
        # grad_E accommodates it.
        r = lambda x: x - langevin_lambda * grad_E(x)


    def langevin_proposal(current_x, preimage_current_x):

        # Refer to our paper for an examplation on the factor 2.0 in there.
        preimage_proposed_x = current_x + np.random.normal(size=current_x.shape, scale=np.sqrt(2*langevin_lambda))
        proposed_x = r(preimage_proposed_x)

        # Now we need to compute
        # log q( current_x | proposed_x ) - log q( proposed_x | current_x )

        asymmetric_correction_log_factor = 0.5/(2*langevin_lambda)*( - ((preimage_current_x - proposed_x)**2).sum() + ((preimage_proposed_x - current_x)**2).sum())

        return (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor)


    def iterate_N_times(current_x, preimage_current_x, energy_difference, N):
        for _ in np.arange(N):
            (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor) = langevin_proposal(current_x, preimage_current_x)

            loga = - energy_difference(proposed_x, current_x) + asymmetric_correction_log_factor
            if accept_all_proposals or loga >= 0 or loga >= np.log(np.random.uniform(0,1)):
                # accepted !
                current_x = proposed_x
                preimage_current_x = preimage_proposed_x
                iterate_N_times.accepted_counter += 1
            else:
                iterate_N_times.rejected_counter += 1

        return (current_x, preimage_current_x)

    iterate_N_times.accepted_counter = 0
    iterate_N_times.rejected_counter = 0


    # Start with the burn-in iterations.
    current_x = x0
    # not quite the actual pre-image, but it's just for initialization purposes
    preimage_current_x = current_x
    (current_x, preimage_current_x) = iterate_N_times(current_x, preimage_current_x, energy_difference, burn_in)

    # Then we can think about collecting samples.
    samples_list = []
    # Start from the 'current_x' from the burn_in
    # and not from x0. Reset the acceptance counters.
    iterate_N_times.accepted_counter = 0
    iterate_N_times.rejected_counter = 0


    for n in np.arange(0,N):
        (current_x, preimage_current_x) = iterate_N_times(current_x, preimage_current_x, energy_difference, thinning_factor)
        # collect sample after running through the thinning iterations
        samples_list.append(current_x)


    samples = np.vstack(samples_list)
    acceptance_ratio = iterate_N_times.accepted_counter * 1.0 / (iterate_N_times.accepted_counter + iterate_N_times.rejected_counter)

    return (samples, acceptance_ratio)







def run_chain_with_SVD(x0, N, thinning_factor = 1, burn_in = 0,
                       E = None, grad_E = None,
                       r = None, r_prime = None, f_prime = None,
                       proposal_stddev = 1.0, accept_all_proposals = False):
    """
        f        g
    X -----> H -----> X

    dim(X) = m
    dim(H) = n
    r = g * f

    In this implementation, we use the following shapes for the arguments.
    r       : R^m -> R^n
    r_prime : R^m -> R^m
    f_prime : R^m -> R^n
    """

    assert len(x0.shape) == 1, "Wrong dimension for x0."
    assert f_prime
    assert r
    assert thinning_factor >= 1, "You misunderstood the thinning_factor. It should be 1 for no thinning, and 32 if we want one out of every 32 samples."

    def approximate_energy_difference(proposed_x, current_x):
        return grad_E(current_x).dot(proposed_x - current_x)

    def exact_energy_difference(proposed_x, current_x):
        return E(proposed_x) - E(current_x)

    def zero_energy_difference(proposed_x, current_x):
        return 0.0

    if not E == None:
        # This would be the regular branch executed when
        # using Monte Carlo.
        energy_difference = exact_energy_difference
    elif E == None and not grad_E == None:
        energy_difference = approximate_energy_difference
    else:
        raise("Unrecognized setup.")

    if r_prime == None:
        r_prime = make_numerical_derivative_function(r)


    def proposal(x_t, previous_time_data = None):
        """
        A lot of quantities have to be carried over from the
        previous iteration. We'll use "tm1" to refer to "t minus 1"
        the previous iteration.
        """

        if not (previous_time_data == None):
            x_tm1 = previous_time_data['x']
            noisy_x_tm1 = previous_time_data['noisy_x']
            logdet_r_prime_noisy_x_tm1 = previous_time_data['logdet_r_prime_noisy_x']
            diagD_tm1 = previous_time_data['diagD']
            Vh_tm1 = previous_time_data['Vh']
            # those two could be recomputed
            inversecov_tm1 = previous_time_data['inversecov']
            logdetcov_tm1 = previous_time_data['logdetcov']


        J_t = f_prime(x_t)

        z = np.random.normal(size=J_t.shape[0])
        #print J_t.shape
        #print z.shape
        noisy_x_t = x_t + proposal_stddev * J_t.T.dot(z)

        (_, diagD_t, Vh_t) = scipy.linalg.svd(J_t)
        inversecov_t = Vh_t.dot(np.diag(1/(proposal_stddev*diagD_t)**2)).dot(Vh_t.T)
        logdetcov_t = 2*np.log(proposal_stddev*diagD_t).sum()

        x_star = r(x_t)
        logdet_r_prime_noisy_x_t = np.log(scipy.linalg.det(r_prime(noisy_x_t)))

        current_time_data = {}
        current_time_data['x'] = x_t
        current_time_data['noisy_x'] = noisy_x_t
        current_time_data['logdet_r_prime_noisy_x'] = logdet_r_prime_noisy_x_t
        current_time_data['diagD'] = diagD_t
        current_time_data['Vh'] = Vh_t
        current_time_data['inversecov'] = inversecov_t
        current_time_data['logdetcov'] = logdetcov_t

        # If we're at the first iteration, we don't care
        # much about the accept / reject because we don't
        # have a pre-image for x_t.
        if previous_time_data == None:
            return (x_star, 0.0, current_time_data)
        else:
            log_q_proposal = 0.5 * (noisy_x_t - x_t).T.dot(inversecov_t).dot(noisy_x_t - x_t) - logdetcov_t - logdet_r_prime_noisy_x_t
            log_q_reverse = 0.5 * (noisy_x_tm1 - x_star).T.dot(inversecov_tm1).dot(noisy_x_tm1 - x_star) - logdetcov_tm1 - logdet_r_prime_noisy_x_tm1

            # We want to return
            #    log q( current_x | proposed_x ) - log q( proposed_x | current_x )
            # which is the quantity of interest to adjust the acceptance ratio.
            asymmetric_correction_log_factor = log_q_reverse - log_q_proposal

            return (x_star, asymmetric_correction_log_factor, current_time_data)
    # end of proposal function

    def iterate_N_times(current_x, previous_time_data, energy_difference, N):
        for _ in np.arange(N):
            (proposed_x, asymmetric_correction_log_factor, current_time_data) = proposal(current_x, previous_time_data)

            loga = - energy_difference(proposed_x, current_x) + asymmetric_correction_log_factor
            if accept_all_proposals or loga >= 0 or loga >= np.log(np.random.uniform(0,1)):
                # accepted !
                current_x = proposed_x
                previous_time_data = current_time_data
                iterate_N_times.accepted_counter += 1
            else:
                iterate_N_times.rejected_counter += 1

        return (current_x, previous_time_data)

    iterate_N_times.accepted_counter = 0
    iterate_N_times.rejected_counter = 0


    # Start with the burn-in iterations.
    current_x = x0
    (current_x, previous_time_data) = iterate_N_times(current_x, None, energy_difference, burn_in)

    # Then we can think about collecting samples.
    samples_list = []
    # Start from the 'current_x' from the burn_in
    # and not from x0. Reset the acceptance counters.
    iterate_N_times.accepted_counter = 0
    iterate_N_times.rejected_counter = 0


    for n in np.arange(0,N):
        (current_x, previous_time_data) = iterate_N_times(current_x, previous_time_data, energy_difference, thinning_factor)
        # collect sample after running through the thinning iterations
        samples_list.append(current_x)


    samples = np.vstack(samples_list)
    acceptance_ratio = iterate_N_times.accepted_counter * 1.0 / (iterate_N_times.accepted_counter + iterate_N_times.rejected_counter)

    return (samples, acceptance_ratio)




def make_numerical_derivative_function(f):
    """
    Takes a function of one argument x of shape (D,).

    Returns function that returns the (D, D) matrix of derivatives at x,
    where the first dimension runs along the output coordinates
    and the second dimension runs along the input variables.

    That is,
       f(x+h) \approx f(x) + f_prime(x) h.
    """
    epsilon = 1.0e-8
    def f_prime(x):
        assert len(x.shape) == 1
        D = x.shape[0]
        deltas = epsilon * np.eye(D)
        res = np.zeros((D,D))
        rx = r(x)
        for d in range(D):
            res[:,d] = (r(x + deltas[d,:]) - rx)/epsilon
        return res
    return f_prime




def get_dict_key_or_default(D, key, default, want_error_if_missing = False):
    if D.has_key(key):
        return D[key]
    else:
        if not want_error_if_missing:
            return default
        else:
            raise("Cannot find key %s in dictionary." % (key,))


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

    #assert not(proposal_stddev == None) and not(langevin_lambda == None)

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


    samples_for_all_chains = np.zeros((n_chains, n_samples, d))
    acceptance_ratio_list = []
    sampling_start_time = time.time()

    for c in np.arange(n_chains):

        if mcmc_method == 'metropolis_hastings_E':
            assert proposal_stddev > 0.0
            symmetric_proposal = lambda x: x + np.random.normal(size=x.shape, scale = proposal_stddev)
            (X, acceptance_ratio) = run_chain_with_energy(E, x0[c,:], symmetric_proposal, n_samples, thinning_factor = thinning_factor, burn_in = burn_in)

        elif mcmc_method == 'metropolis_hastings_grad_E':
            assert proposal_stddev > 0.0
            symmetric_proposal = lambda x: x + np.random.normal(size=x.shape, scale = proposal_stddev)
            (X, acceptance_ratio) = run_chain_with_energy(None, x0[c,:], symmetric_proposal, n_samples, thinning_factor = thinning_factor, burn_in = burn_in, grad_E = grad_E)

        elif mcmc_method == 'langevin':
            (X, acceptance_ratio) = run_chain_with_langevin_proposals(x0[c,:], n_samples, langevin_lambda, grad_E = grad_E, thinning_factor = thinning_factor, burn_in = burn_in, accept_all_proposals = True)

        elif mcmc_method == 'metropolis_hastings_langevin_E':

            # grad_E is needed to define the perfect reconstruction function
            # E is used for the Metropolis-Hastings sampling

            # remember that we have a minus here because it's d log p(x) / dx
            # which corresponds to - dE(x) / dx
            r = lambda x: x - langevin_lambda * grad_E(x)
            (X, acceptance_ratio) = run_chain_with_langevin_proposals(x0[c,:], n_samples, langevin_lambda, E = E, thinning_factor = thinning_factor, burn_in = burn_in, r = r)

            #(X, acceptance_ratio) = metropolis_hastings_sampler.run_chain_with_energy(ninja_star_distribution.E, x0, None, n_samples, thinning_factor = thinning_factor, burn_in = burn_in, asymmetric_proposal = asymmetric_proposal)

        elif mcmc_method == 'metropolis_hastings_langevin_grad_E':

            (X, acceptance_ratio) = run_chain_with_langevin_proposals(x0[c,:], n_samples, langevin_lambda, grad_E = grad_E, thinning_factor = thinning_factor, burn_in = burn_in)

            #(asymmetric_proposal, r) = metropolis_hastings_sampler.make_langevin_sampler_requirements(langevin_lambda, ninja_star_distribution.grad_E)
            #(X, acceptance_ratio) = metropolis_hastings_sampler.run_chain_with_energy(None, x0, None, n_samples, thinning_factor = thinning_factor, burn_in = burn_in, grad_E = ninja_star_distribution.grad_E, asymmetric_proposal = asymmetric_proposal)

        elif mcmc_method == "metropolis_hastings_svd_grad_E":

            (X, acceptance_ratio) = run_chain_with_SVD(x0[c,:], n_samples, proposal_stddev = proposal_stddev, grad_E = grad_E, r = r, r_prime = r_prime, f_prime = f_prime, thinning_factor = thinning_factor, burn_in = burn_in)

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


