import numpy as np
import time
import scipy

######################################################################
# This implementation is broken because it fails to compensate
# for the densities in the jacobians. Anyways, it's a TODO.
# We have to review this method and tweak it before assuming that
# it outputs samples with the correct asymptotic distribution.
##############

def sample_chain(x0, N,
                 energy_difference, proposal_stddev,
                 thinning_factor = 1, burn_in = 0):
    """
    Vanilla Monte Carlo Markov Chain that proposes changes
    according to isotropic a Normal distribution N(0, proposal_stddev).

    This makes absolutely no use of the fact that we
    are dealing with an autoencoder apart from the
    fact that the energy_difference function is usually
    meant to be obtained from a DAE's reconstruction function.
    """

    if len(x0.shape) != 1:
        error("Wrong dimension for x0. This function is not vectorial.")

    if thinning_factor < 1:
        error("You misunderstood the thinning_factor. It should be 1 for no thinning, and 32 if we want one out of every 32 samples.")

    proposal = lambda current_x: current_x + np.random.normal(size=current_x.shape, scale=proposal_stddev)


    def iterate_N_times(current_x, energy_difference, N):
        for _ in np.arange(N):
            proposed_x = proposal(current_x)
            loga = - energy_difference(proposed_x, current_x)
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
