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
                 accept_all_proposals = False):
    """
    Will sample N values for the chain starting with x0.

    noise_levels is a dict with keys 
    ["train_stddev"], ["train_stddev", "langevin_beta"] or ["train_stddev", "langevin_stddev"]

    """

    assert len(x0.shape) == 1, "Wrong dimension for x0."

    assert thinning_factor >= 1, "You misunderstood the thinning_factor. It should be 1 for no thinning, and 32 if we want one out of every 32 samples."

    assert noise_levels.has_key("train_stddev")
    train_stddev = noise_levels["train_stddev"]

    if noise_levels.has_key("langevin_stddev") and noise_levels.has_key("langevin_beta"):
        langevin_stddev = noise_levels["langevin_stddev"]
        langevin_beta = noise_levels["langevin_beta"]

    elif noise_levels.has_key("langevin_stddev") and not noise_levels.has_key("langevin_beta"):
        langevin_stddev = noise_levels["langevin_stddev"]
        temperature = 1.0
        langevin_beta = langevin_stddev**2 / (2*temperature*train_stddev**2)

    elif noise_levels.has_key("langevin_beta") and not noise_levels.has_key("langevin_stddev"):
        langevin_beta = noise_levels["langevin_beta"]
        temperature = 1.0
        langevin_stddev = np.sqrt(2*temperature*langevin_beta)*train_stddev

    else:
        # we've got nothing, so let's pick beta=1
        langevin_beta = 1.0
        temperature = 1.0
        langevin_stddev = np.sqrt(2*temperature*langevin_beta)*train_stddev

    assert train_stddev > 0
    assert langevin_stddev > 0
    assert not (langevin_beta == 0.0)
    if langevin_beta < 0.0 or 1.0 < langevin_beta:
        print "This is not **NECESSARILY** an error, but it is a bit strange to be using a beta outside of the range [0,1]."

    temperature = langevin_stddev**2 / ( 2 * langevin_beta * train_stddev**2 )
    print "==========================="
    print "With your current setup, you have that the sampling procedure scales as follows."
    print ""
    print "train_stddev : %f" % train_stddev
    print "langevin_stddev : %f" % langevin_stddev
    print 
    print "langevin_beta : %f" % langevin_beta
    print "temperature : %f" % temperature
    print "==========================="
    noise_levels = {'train_stddev':train_stddev,
                    'langevin_stddev':langevin_stddev,
                    'langevin_beta':langevin_beta,
                    'temperature':temperature}

    def langevin_proposal(current_x, preimage_current_x, want_proposal_log_ratio=True):

        # We are using the term "preimage" here because it corresponds
        # to the preimage when langevin_beta=1.0.
        # Otherwise, it should be called the "noisy_ancestor" or something
        # like that to reflect the fact that it's more about
        #
        # x_{\textrm{noisy}}^{(t)}&=&x^{(t)}+\epsilon\hspace{1em}for\hspace{1em}\epsilon\sim\mathcal{N}(0,\sigma^{2})
        # x^{*}&=&\left(1-\beta\right)x_{\textrm{noisy}}^{(t)}+\beta r^{*}(x_{\textrm{noisy}}^{(t)})
        #
        # than about being the preimage. Latex the stuff above to read it properly.

        d = current_x.shape[0]
        preimage_proposed_x = current_x + np.random.normal(size=(d,), scale=langevin_stddev)
        proposed_x = (1-langevin_beta) * preimage_proposed_x + langevin_beta * r(preimage_proposed_x)

        if want_proposal_log_ratio:
            # Now we need to compute
            # log q( current_x | proposed_x ) - log q( proposed_x | current_x )

            A = np.zeros((2,))
            A[0] = - 0.5/langevin_stddev**2 * ((preimage_current_x - proposed_x)**2).sum()
            A[1] = -1 * np.linalg.det( (1-langevin_beta) * np.eye(d) +  langevin_beta * r_prime(preimage_current_x))

            B = np.zeros((2,))
            B[0] = - 0.5/langevin_stddev**2 * ((preimage_proposed_x - current_x)**2).sum()
            B[1] = -1 * np.linalg.det( (1-langevin_beta) * np.eye(d) +  langevin_beta * r_prime(preimage_proposed_x))

            asymmetric_correction_log_factor = A[0] + A[1] - B[0] - B[1]
        else:
            asymmetric_correction_log_factor = 0.0

        return (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor)


    def iterate_N_times(current_x, preimage_current_x, energy_difference, N):
        for _ in np.arange(N):
            (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor) = langevin_proposal(current_x, preimage_current_x, not(accept_all_proposals))

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

    progress_logger = make_progress_logger("Sampling")

    for n in np.arange(0,N):
        (current_x, preimage_current_x) = iterate_N_times(current_x, preimage_current_x, energy_difference, thinning_factor)
        # collect sample after running through the thinning iterations
        samples_list.append(current_x)
        progress_logger(1.0*n/N)

    samples = np.vstack(samples_list)
    acceptance_ratio = iterate_N_times.accepted_counter * 1.0 / (iterate_N_times.accepted_counter + iterate_N_times.rejected_counter)

    return (samples, acceptance_ratio, noise_levels)