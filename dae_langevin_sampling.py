
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


def plot_spiral_into_axes(axes):
    from matplotlib.lines import Line2D
    import debian_spiral
    N = 80
    spiral_samples = debian_spiral.sample(N, 0.0, want_sorted_data = True, want_evenly_spaced = True)
    for i in range(N-1):
        l = Line2D([spiral_samples[i,0],spiral_samples[i+1,0]],
                   [spiral_samples[i,1],spiral_samples[i+1,1]],
                   linestyle='--', linewidth=1.0, c='#f9a21d')
        axes.add_line(l)


def run_langevin_simulation(mydae, simulated_samples, noise_stddev, n_iter, n_sub_iter, noise_method = 'iso_x'):

    # The noise_stddev should be something like
    #
    #    sqrt((d+2)/2) * train_noise_stddev
    #
    # according to what I read from the most recent
    # copy of our paper that I can find lying around my desk.

    # Acceptable values for noise_method include:
    # 'iso_x', 'JTJ_h', 'iso_h'

    logged_simulated_samples = np.zeros((n_iter, simulated_samples.shape[0], simulated_samples.shape[1]))

    for i in range(n_iter):
        logged_simulated_samples[i,:,:] = simulated_samples
        print("iteration %d done" % i)
        for _ in range(n_sub_iter):

            if noise_method == 'iso_x':
                # isotropic noise added to every sample
                simulated_samples = simulated_samples + np.random.normal(scale = noise_stddev, size = simulated_samples.shape)
                # encode_decode is conveniently constructed as a vectored
                # function along the 0-th dimension
                simulated_samples = mydae.encode_decode(simulated_samples)
            elif noise_method == 'iso_h':
                simulated_samples_h = mydae.encode(simulated_samples)
                simulated_samples_h = simulated_samples_h + np.random.normal(scale = noise_stddev, size = simulated_samples_h.shape)
                simulated_samples = mydae.decode(simulated_samples_h)
            elif noise_method == 'JTJ_h':
                # unfortunately, we don't have a vectorized version of this code for the jacobian of the encoder
                simulated_samples_h = mydae.encode(simulated_samples)
                for k in range(simulated_samples_h.shape[0]):
                    # Compute the jacobian J, draw from a normal, matrix-multiply by J
                    # and that gets you a normal drawn from covariance J^TJ.
                    # We still have the noise_stddev in there even though we're
                    # no longer totally sure where it should really be.
                    J = mydae.encoder_jacobian_single(simulated_samples[k,:])
                    # print J.shape
                    # print simulated_samples.shape
                    #
                    # So it's J multiplying normal noise in the space of dimension of X ?
                    e = J.dot(np.random.normal(scale = noise_stddev, size = (simulated_samples.shape[1],1)))
                    # print e.shape
                    simulated_samples_h[k,:] = simulated_samples_h[k,:] + e.reshape((-1,))


                simulated_samples = mydae.decode(simulated_samples_h)
            else:
                error('noise_method not recognized : ' + noise_method)

    return logged_simulated_samples


def write_simulated_samples_frames(logged_simulated_samples, filename_generator_function, window_width=1.0, center=(0.0, 0.0)):

    for i in range(logged_simulated_samples.shape[0]):

        outputfile = filename_generator_function(i)

        pylab.hold(True)
        plot_spiral_into_axes(pylab.gca())
        pylab.scatter(logged_simulated_samples[i,:,0],
                      logged_simulated_samples[i,:,1],
                      c='#c20aab')
        pylab.axis([center[0] - window_width*1.0, center[0] + window_width*1.0,
                    center[1] - window_width*1.0, center[1] + window_width*1.0])
        pylab.draw()
        pylab.savefig(outputfile, dpi=100)
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

    n_iter = 100
    n_sub_iter = 100

    # should check if we have to multiply by sqrt(2)=1.414

    logged_simulated_samples = run_langevin_simulation(mydae, simulated_samples, train_noise_stddev * 1.414, n_iter, n_sub_iter, noise_method='JTJ_h')

    write_simulated_samples_frames(logged_simulated_samples,
                                   lambda i: os.path.join(simulated_frames_output_dir, "simulation_frame_%0.5d.png" % i),
                                   window_width=1.0)

    # Save the results for future use.
    cPickle.dump(logged_simulated_samples, open(os.path.join(simulated_frames_output_dir, "logged_simulated_samples.pkl"), "w"))

    # Now we'll generate a collection of possibly useful plots for the ICLR paper.
    # Maybe it would be better to skip the first half of the points, but
    # in practice things stabilize rather fast so it might not be necessary.

    #combined_logged_simulated_samples = np.vcat([logged_simulated_samples[i,j,:].reshape((1,-1))
    #                                             for i in range logged_simulated_samples.shape[0]
    #                                             for j in range logged_simulated_samples.shape[1]])

    combined_logged_simulated_samples = logged_simulated_samples.reshape((1,-1,logged_simulated_samples.shape[2]))
    write_simulated_samples_frames(combined_logged_simulated_samples,
                                   lambda i: os.path.join(simulated_frames_output_dir, "all_frames_width_0.5.png"),
                                   window_width=0.5)
    write_simulated_samples_frames(combined_logged_simulated_samples,
                                   lambda i: os.path.join(simulated_frames_output_dir, "all_frames_width_1.0.png"),
                                   window_width=1.0)
    write_simulated_samples_frames(combined_logged_simulated_samples,
                                   lambda i: os.path.join(simulated_frames_output_dir, "all_frames_width_2.0.png"),
                                   window_width=2.0)


if __name__ == "__main__":
    main()




def plot_reference_spiral_for_ICLR_paper(simulated_frames_output_dir, n_simulated_samples, n_iter):
    # Plot the reference graph for the ICLR paper.
    import debian_spiral
    original_spiral_samples = debian_spiral.sample(n_simulated_samples * n_iter, 0.0).reshape((1,-1,2))
    write_simulated_samples_frames(original_spiral_samples,
                                   lambda i: os.path.join(simulated_frames_output_dir, "original_spiral_samples_width_0.5.png"),
                                   window_width=0.5)
    write_simulated_samples_frames(original_spiral_samples,
                                   lambda i: os.path.join(simulated_frames_output_dir, "original_spiral_samples_width_1.0.png"),
                                   window_width=1.0)
    write_simulated_samples_frames(original_spiral_samples,
                                   lambda i: os.path.join(simulated_frames_output_dir, "original_spiral_samples_width_2.0.png"),
                                   window_width=2.0)


# plot_reference_spiral_for_ICLR_paper("/u/alaingui/Documents/tmp/14", 100, 100)



#  python dae_langevin_sampling.py $HOME/umontreal/denoising_autoencoder/plots/experiment_970584/results.pkl $HOME/Documents/tmp/1

# ffmpeg -i /u/alaingui/Documents/tmp/4/simulation_frame_%05d.png -vcodec mpeg4 /u/alaingui/Documents/tmp/4/sequence.avi