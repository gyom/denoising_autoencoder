import numpy as np
import time
import scipy

import refactor_gp
import refactor_gp.gyom_utils
from refactor_gp.gyom_utils import make_progress_logger


def sample_chain(x0, N,
                 energy_difference, noise_levels,
                 r, r_prime, f_prime,
                 thinning_factor = 1, burn_in = 0,
                 accept_all_proposals = False):
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

    noise_levels is a dict with keys 
    ["train_stddev"], ["train_stddev", "langevin_beta"] or ["train_stddev", "langevin_stddev"]
    """

    assert len(x0.shape) == 1, "Wrong dimension for x0."
    assert f_prime

    assert noise_levels.has_key("train_stddev")
    train_stddev = noise_levels["train_stddev"]

    if noise_levels.has_key("langevin_stddev") and noise_levels.has_key("langevin_beta"):
        print "You should NOT specify both langevin_stddev and langevin_beta, but we'll accept this only if the values are compatible."
        langevin_stddev = noise_levels["langevin_stddev"]
        langevin_beta = noise_levels["langevin_beta"]
        assert langevin_stddev == train_stddev**2 / (2*langevin_beta)

    elif noise_levels.has_key("langevin_stddev") and not noise_levels.has_key("langevin_beta"):
        langevin_stddev = noise_levels["langevin_stddev"]
        langevin_beta = train_stddev**2 / (2*langevin_stddev**2)

    elif noise_levels.has_key("langevin_beta") and not noise_levels.has_key("langevin_stddev"):
        langevin_beta = noise_levels["langevin_beta"]
        langevin_stddev = train_stddev**2 / (2*langevin_beta)

    else:
        # we've got nothing, so let's pick beta=1
        langevin_beta = 1.0
        langevin_stddev = train_stddev**2 / (2*langevin_beta)

    assert train_stddev > 0
    assert langevin_stddev > 0
    assert not (langevin_beta == 0.0)
    if langevin_beta < 0.0 or 1.0 < langevin_beta:
        print "This is not necessarily an error, but it is a bit strange to be using a langevin_beta outside of the range [0,1]."


    def proposal(current_x, preimage_current_x):

        d = current_x.shape[0]

        J = f_prime(current_x) * langevin_stddev
        det_JTJ = np.linalg.det(J.T.dot(J))
        z = np.random.normal(size=J.shape[0])
        preimage_proposed_x = current_x + J.T.dot(z)
        proposed_x = (1-langevin_beta) * current_x + langevin_beta * r(preimage_proposed_x)
        proposed_J = f_prime(proposed_x)
        det_proposed_JTJ = np.linalg.det(proposed_J.T.dot(proposed_J))

        # Bear in mind that the covariance of the mvn stemming from current_x
        # will be J^T J and not just J.
        assert J.shape[1] == d
        assert proposed_J.shape[1] == d

        # We will essentially bypass the SVD decomposition by
        # using J^T J instead of V^T D^2 V from the SVD.
        # The two quantities are equivalent.
        # It would still be nice, in a way, to have access to the eigenvalues
        # in order to have more control (truncating ?) and be able to log them
        # as some kind of sanity check (to check Yoshua's fast decay intuition).

        # Now we need to compute
        # log q( current_x | proposed_x ) - log q( proposed_x | current_x )

        A = np.zeros((2,))
        v = (preimage_current_x - proposed_x)
        A[0] = - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(det_JTJ) - 0.5 * v.dot(np.linalg.inv(J.T.dot(J))).dot(v)
        A[1] = -1 * np.linalg.det( (1-langevin_beta) * np.eye(d) +  langevin_beta * r_prime(preimage_current_x))

        B = np.zeros((2,))
        v = (preimage_proposed_x - current_x)
        B[0] = - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(det_proposed_JTJ) - 0.5 * v.dot(np.linalg.inv(proposed_J.T.dot(proposed_J))).dot(v)
        B[1] = -1 * np.linalg.det( (1-langevin_beta) * np.eye(d) +  langevin_beta * r_prime(preimage_proposed_x))

        asymmetric_correction_log_factor = A[0] + A[1] - B[0] - B[1]

        return (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor)


    # end of proposal function

    def iterate_N_times(current_x, preimage_current_x, energy_difference, N):
        for _ in np.arange(N):
            (proposed_x, preimage_proposed_x, asymmetric_correction_log_factor) = proposal(current_x, preimage_current_x)

            # This is a - in front of the energy difference because
            # log( p(proposed_x) / p(current_x) ) \approx -E(proposed_x) - -E(current_x) = - energy_difference(proposed_x, current_x)
            loga = - energy_difference(proposed_x, current_x) + asymmetric_correction_log_factor
            if accept_all_proposals or loga >= 0 or loga >= np.log(np.random.uniform(0,1)):
                # accepted !
                current_x = proposed_x
                preimage_current_x = preimage_proposed_x
                iterate_N_times.accepted_counter += 1

                # DEBUG
                #print "Accepted transition with loga = %0.2f" % loga
                #print proposed_x
            else:
                iterate_N_times.rejected_counter += 1

                # DEBUG
                #print "Rejected transition with loga = %0.2f" % loga
                #print proposed_x


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

    return (samples, acceptance_ratio)

