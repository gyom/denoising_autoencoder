
import numpy as np

def run_chain_with_energy(E, x0, symmetric_proposal, N, thinning_factor = 1, burn_in = 0):
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

    # Same algorithm as below, but running for burn_in.
    # It's a bit of code duplication.
    current_x = x0
    current_E = E(current_x)
    for _ in np.arange(0,burn_in):
        proposed_x = symmetric_proposal(current_x)
        loga = -E(proposed_x) + current_E
        if loga >= 0 or loga >= np.log(np.random.uniform(0,1)):
            current_x = proposed_x
            current_E = E(current_x)


    d = x0.shape[0]
    samples = np.zeros((N,d))
    # Start from the 'current_x' from the burn_in
    # and not from x0.
    samples[0,:] = current_x

    accepted_counter = 0
    rejected_counter = 0

    for n in np.arange(0,N-1):
        current_x = samples[n,:]
        # cache the energy to avoid recomputing
        current_E = E(current_x)

        for i in np.arange(0,thinning_factor):
            proposed_x = symmetric_proposal(current_x)
            loga = -E(proposed_x) + current_E
            #print "loga = %f" % loga
            if loga >= 0 or loga >= np.log(np.random.uniform(0,1)):
                # accepted !
                current_x = proposed_x
                current_E = E(current_x)
                accepted_counter = accepted_counter + 1
            else:
                rejected_counter = rejected_counter + 1

        samples[n+1,:] = current_x

    return (samples,
            accepted_counter * 1.0 / (accepted_counter + rejected_counter) )
