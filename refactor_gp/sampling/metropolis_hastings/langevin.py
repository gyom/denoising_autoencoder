import numpy as np
import time
import scipy

def sample_chain(x0, N,
                energy_difference, langevin_lambda,
                r, r_prime,
                thinning_factor = 1, burn_in = 0,
                accept_all_proposals = False):
    """
    Will sample N values for the chain starting with x0.
    """

    assert len(x0.shape) == 1, "Wrong dimension for x0."

    assert thinning_factor >= 1, "You misunderstood the thinning_factor. It should be 1 for no thinning, and 32 if we want one out of every 32 samples."
    assert langevin_lambda > 0.0, "It doesn't make sense to have the langevin lambda be less than 0. With 0.0, a lot becomes pointless."


    def langevin_proposal(current_x, preimage_current_x):

        # Refer to our paper for an examplation on the factor 2.0 in there.
        preimage_proposed_x = current_x + np.random.normal(size=current_x.shape, scale=np.sqrt(2*langevin_lambda))
        proposed_x = r(preimage_proposed_x)

        # Now we need to compute
        # log q( current_x | proposed_x ) - log q( proposed_x | current_x )

        asymmetric_correction_log_factor = ( (0.5/(2*langevin_lambda)*( - ((preimage_current_x - proposed_x)**2).sum() +
                                                                          ((preimage_proposed_x - current_x)**2).sum()))  +
                                             -1 * np.linalg.det(r_prime(preimage_current_x)) +
                                              1 * np.linalg.det(r_prime(preimage_proposed_x))   )

        return (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor)


    def iterate_N_times(current_x, preimage_current_x, energy_difference, N):
        for _ in np.arange(N):
            (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor) = langevin_proposal(current_x, preimage_current_x)

            # This is a - in front of the energy difference because
            # log( p(proposed_x) / p(current_x) ) \approx -E(proposed_x) - -E(current_x) = - energy_difference(proposed_x, current_x)
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