import numpy as np
import time
import scipy

######################################################################
# This implementation is broken because it fails to compensate
# for the densities in the jacobians. Anyways, it's a TODO.
# We have to review this method and tweak it before assuming that
# it outputs samples with the correct asymptotic distribution.
##############

def sample_chain(E, x0, symmetric_proposal, N, thinning_factor = 1, burn_in = 0, grad_E = None, asymmetric_proposal = None):
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
