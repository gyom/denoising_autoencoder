
import numpy as np

def run_chain_with_energy(E, x0, symmetric_proposal, N, thinning_factor = 1, burn_in = 0, grad_E = None):
    """
    Will sample N values for the chain starting with x0.
    The energy function is given by E(...) which takes a
    vector similar to x0 as argument.

    'symmetric_proposal' is a function that yields the
    next position proposed. It is assumed to be symmetrical
    because we don't compensate in the ratio.
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



    def iterate_N_times(current_x, energy_difference, N):
        for _ in np.arange(N):
            proposed_x = symmetric_proposal(current_x)
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
