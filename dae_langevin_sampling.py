
import numpy as np
import cPickle
import os
import sys

import dae

import matplotlib
matplotlib.use('Agg')
import pylab



def get_starting_point_for_spiral():
    import debian_spiral
    spiral_samples = debian_spiral.sample(1000, 0.0, want_sorted_data = True)
    return spiral_samples[0,:]


def run_langevin_simulation(mydae, simulated_samples, noise_stddev, n_iter, n_sub_iter):

    # The noise_stddev should be something like
    #
    #    sqrt((d+2)/2) * train_noise_stddev
    #
    # according to what I read from the most recent
    # copy of our paper that I can find lying around my desk.

    logged_simulated_samples = np.zeros((n_iter, simulated_samples.shape[0], simulated_samples.shape[1]))

    for i in range(n_iter):
        logged_simulated_samples[i,:,:] = simulated_samples
        print("iteration %d done" % i)
        for _ in range(n_sub_iter):
            # isotropic noise added to every sample
            simulated_samples = simulated_samples + np.random.normal(scale = noise_stddev, size = simulated_samples.shape)
            # encode_decode is conveniently constructed as a vectored
            # function along the 0-th dimension
            simulated_samples = mydae.encode_decode(simulated_samples)

    return logged_simulated_samples


def write_simulated_samples_frames(logged_simulated_samples, filename_generator_function):

    for i in range(logged_simulated_samples.shape[0]):

        outputfile = filename_generator_function(i)

        pylab.hold(True)
        pylab.scatter(logged_simulated_samples[i,:,0],
                      logged_simulated_samples[i,:,1],
                      c='#c20aab')
        pylab.draw()
        pylab.savefig(outputfile, dpi=300)
        pylab.close()
        print("Wrote " + outputfile)

def main():

    if len(sys.argv) < 3:
        error("You need two arguments. The pickled results file, and the output directory for the simulated frames.")

    pickled_results_file = sys.argv[1]
    simulated_frames_output_dir = sys.argv[2]

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
    n_sub_iter = 1000
    logged_simulated_samples = run_langevin_simulation(mydae, simulated_samples, train_noise_stddev * 1.414, n_iter, n_sub_iter)

    write_simulated_samples_frames(logged_simulated_samples,
                                   lambda i: os.path.join(simulated_frames_output_dir, "simulation_frame_%0.5d" % i))




if __name__ == "__main__":
    main()



#  python dae_langevin_sampling.py $HOME/umontreal/denoising_autoencoder/plots/experiment_970584/results.pkl $HOME/Documents/tmp/1