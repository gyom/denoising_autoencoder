import numpy as np
import time
import scipy

def sample_chain(x0, N,
                energy_difference,
                thinning_factor = 1, burn_in = 0,
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
    
    energy_difference : (R^m, R^m) -> R
                        proposed_x, current_x   |->   log(p(proposed_x)) - log(p(current_x))
    
    """

    assert len(x0.shape) == 1, "Wrong dimension for x0."
    assert f_prime
    assert r

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
