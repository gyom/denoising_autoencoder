import numpy as np
import time
import scipy

import refactor_gp
import refactor_gp.gyom_utils
from refactor_gp.gyom_utils import make_progress_logger


def sample_chain(x0, N,
                 energy_difference, noise_levels,
                 r, r_prime,
                 thinning_factor = 1, burn_in = 0,
                 accept_all_proposals = False, proposal_noise_scheme = 'merge_x', omit_asymmetric_proposal_factor = False):
    """
    Will sample N values for the chain starting with x0.

    noise_levels is a dict with keys 
    ["train_stddev"], ["train_stddev", "langevin_beta"] or ["train_stddev", "langevin_stddev"]

    """

    print proposal_noise_scheme

    assert len(x0.shape) == 1, "Wrong dimension for x0."

    assert thinning_factor >= 1, "You misunderstood the thinning_factor. It should be 1 for no thinning, and 32 if we want one out of every 32 samples."

    train_stddev    = noise_levels["train_stddev"]
    langevin_stddev = noise_levels["langevin_stddev"]
    langevin_beta   = noise_levels["langevin_beta"]
    temperature     = noise_levels["temperature"]

    def langevin_proposal(current_x, preimage_current_x):

        # We are using the term "preimage" here because it corresponds
        # to the preimage when langevin_beta=1.0.
        # Otherwise, it should be called the "noisy_ancestor" or something
        # like that to reflect the fact that it's more about
        #
        # x_{\textrm{noisy}}^{(t)}&=&x^{(t)}+\epsilon\hspace{1em}for\hspace{1em}\epsilon\sim\mathcal{N}(0,\sigma^{2})
        # x^{*}&=&\left(1-\beta\right)x_{\textrm{noisy}}^{(t)}+\beta r^{*}(x_{\textrm{noisy}}^{(t)})
        #
        # than about being the preimage. Latex the stuff above to read it properly.

        # This function accesses the variables from the "closure" : accept_all_proposals, proposal_noise_scheme

        d = current_x.shape[0]

        if proposal_noise_scheme == 'merge_x':
            preimage_proposed_x = current_x + np.random.normal(size=(d,), scale=langevin_stddev)
            proposed_x = (1-langevin_beta) * preimage_proposed_x + langevin_beta * r(preimage_proposed_x)
        elif proposal_noise_scheme == 'noise_E':
            preimage_proposed_x = current_x + np.random.normal(size=(d,), scale=langevin_stddev)
            proposed_x = current_x - langevin_beta * preimage_proposed_x + langevin_beta * r(preimage_proposed_x)
        elif proposal_noise_scheme == 'noise_r':
            preimage_proposed_x = current_x + np.random.normal(size=(d,), scale=langevin_stddev)
            proposed_x = (1-langevin_beta)*current_x  + langevin_beta * r(preimage_proposed_x)
        else:
            raise("Unrecognized proposal_noise_scheme : %s" % proposal_noise_scheme)

        if accept_all_proposals or omit_asymmetric_proposal_factor:
            asymmetric_correction_log_factor = 0.0
        else:
            # Now we need to compute
            # log q( current_x | proposed_x ) - log q( proposed_x | current_x )

            A = np.zeros((2,))
            B = np.zeros((2,))

            A[0] = - 0.5/langevin_stddev**2 * ((preimage_current_x - proposed_x)**2).sum()
            B[0] = - 0.5/langevin_stddev**2 * ((preimage_proposed_x - current_x)**2).sum()
            if proposal_noise_scheme == 'merge_x':
                A[1] = -1 * np.log( np.linalg.det( (1-langevin_beta) * np.eye(d) +  langevin_beta * r_prime(preimage_current_x)) )
                B[1] = -1 * np.log( np.linalg.det( (1-langevin_beta) * np.eye(d) +  langevin_beta * r_prime(preimage_proposed_x)) )
            elif proposal_noise_scheme == 'noise_E':
                # clueless
                A[1] = -1 * np.log( np.linalg.det( (-langevin_beta) * np.eye(d) + langevin_beta * r_prime(preimage_current_x)) )
                B[1] = -1 * np.log( np.linalg.det( (-langevin_beta) * np.eye(d) + langevin_beta * r_prime(preimage_proposed_x)) )
                #pass
            elif proposal_noise_scheme == 'noise_r':
                # clueless
                A[1] = -1 * np.log(  np.linalg.det( langevin_beta * r_prime(preimage_current_x)) )
                B[1] = -1 * np.log( np.linalg.det( langevin_beta * r_prime(preimage_proposed_x)) )
                #pass
            else:
                raise("Unrecognized proposal_noise_scheme : %s" % proposal_noise_scheme)

            asymmetric_correction_log_factor = A[0] + A[1] - B[0] - B[1]                


        return (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor)


    def iterate_N_times(current_x, preimage_current_x, energy_difference, N):
        for _ in np.arange(N):
            (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor) = langevin_proposal(current_x, preimage_current_x)

            if accept_all_proposals:
                loga = 0.0
            else:
                # This is a - in front of the energy difference because
                # log( p(proposed_x) / p(current_x) ) \approx -E(proposed_x) - -E(current_x) = - energy_difference(proposed_x, current_x)
                loga = - energy_difference(proposed_x, current_x) / temperature + asymmetric_correction_log_factor
                # loga = - energy_difference(proposed_x, current_x) + asymmetric_correction_log_factor

            if accept_all_proposals or loga >= 0.0 or loga >= np.log(np.random.uniform(0,1)):
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

    progress_logger = make_progress_logger("Sampling")

    for n in np.arange(0,N):
        (current_x, preimage_current_x) = iterate_N_times(current_x, preimage_current_x, energy_difference, thinning_factor)
        # collect sample after running through the thinning iterations
        samples_list.append(current_x)
        progress_logger(1.0*n/N)

    samples = np.vstack(samples_list)
    acceptance_ratio = iterate_N_times.accepted_counter * 1.0 / (iterate_N_times.accepted_counter + iterate_N_times.rejected_counter)

    return (samples, acceptance_ratio, noise_levels)