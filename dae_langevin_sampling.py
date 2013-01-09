
import numpy as np
import cPickle
import os
import sys

import dae



def get_starting_point_for_spiral():
    import debian_spiral
    spiral_samples =debian_spiral.sample(1000, 0.0, want_sorted_data = True)
    return spiral_samples[0,:]


def run_langevin_simulation(mydae, simulated_samples, noise_stddev, n_iter):

    # The noise_stddev should be something like
    #
    #    sqrt((d+2)/2) * train_noise_stddev
    #
    # according to what I read from the most recent
    # copy of our paper that I can find lying around my desk.

    #for _ in range(n_iter):
    #    simulated_samples = 1

    pass

def main():

    pickled_results_file = sys.argv[1]
    E = cPickle.load(open(pickled_results_file, "r"))

    for k in ['Wb', 'Wc', 'b', 'c', 'n_inputs', 'n_hiddens', 'output_scaling_factor', 'train_noise_stddev']:
        if not k in E.keys():
            error("Missing key %s from the pickled file %s." % (k, pickled_results_file) )

    n_inputs  = E['n_inputs']
    n_hiddens = E['n_hiddens']

    if (n_inputs, n_hiddens) != E['Wb'].shape:
        error("Wrong shape for Wb.")

    if (n_inputs, n_hiddens) != E['Wc'].shape:
        error("Wrong shape for  Wc.")

    mydae = dae.DAE(n_inputs = n_inputs,
                    n_hiddens = n_hiddens,
                    output_scaling_factor = E['output_scaling_factor'])
    mydae.Wb = E['Wb']
    mydae.Wc = E['Wc']
    mydae.b  = E['b']
    mydae.c  = E['c']

    train_noise_stddev = E['train_noise_stddev']

    # We start all the samples at the same place on the spiral.
    n_simulated_samples = 100
    simulated_samples = np.tile(get_starting_point_for_spiral().reshape((1,-1)), (n_simulated_samples, 1))

    n_iter = 1000
    run_langevin_simulation(mydae, simulated_samples, train_noise_stddev * 1.414, n_iter)



if __name__ == "__main__":
    main()



#  python dae_langevin_sampling.py $HOME/umontreal/denoising_autoencoder/plots/experiment_695107/results.pkl