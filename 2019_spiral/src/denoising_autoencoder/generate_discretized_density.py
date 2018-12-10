"""
One thing that will come in handy later will be to have access to a good
approximation of the density p(x) of the spiral. For that purpose, we will
precompute a grid with a reasonably fine resolution and we will estimate very
accurately p(x).

The output of this depends on the grid chosen, and of the amount of "natural noise"
used to define the thickness of our spiral.
"""


import sys
sys.path.insert(0,'/current_project/src')

import os
import pickle

import numpy as np
import scipy
import scipy.stats

# Using tensorflow to handle parameters. This is almost silly, but we have
# tensorflow anyways for other things so we might as well use it here too.
import tensorflow as tf

import denoising_autoencoder
import denoising_autoencoder.spiral

from denoising_autoencoder.logsumexp import logsumexp

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "output_pickle_path", None,
    "Pickle file where we want to store the computed density.")

flags.DEFINE_integer(
    "grid_nbr_points", 1000,
    "Generate a square grid of shape (2*grid_nbr_points+1, 2*grid_nbr_points+1 ).")

flags.DEFINE_float(
    "grid_radius", 4.0,
    "Generate a square grid spanning [-grid_radius, grid_radius]^2.")

flags.DEFINE_float(
    "spiral_noise_sigma", 0.01,
    "Defines the intrinsic width of the spiral.")

flags.DEFINE_integer(
    "nbr_iter", 100,
    "Number of samples to draw and spread their weights.")

FLAGS = flags.FLAGS


def normal_pdf_contributions_on_grid(x, y, grid_x, grid_y, sigma, want_log=False):
    """
    We got a point (x, y) from the spiral, and we want to spread that
    probability mass as a gaussian around that point.

    We are working with a discretized grid so we only want to add
    the contributions to that grid.

    We also need to make sure that the contributions INTEGRATE to 1.0,
    which is trickier than just normalizing them because we need to
    take into account the grid itself.
    """

    # check that this is the right orientation
    delta_x = (x - grid_x) / sigma
    delta_y = (y - grid_y) / sigma

    # We will flatten the thing

    d2 = delta_x**2 + delta_y**2
    d2 = d2.reshape((-1,))

    if want_log:
        pdf_values = scipy.stats.norm.logpdf(d2, loc=0.0, scale=1.0)
    else:
        pdf_values = scipy.stats.norm.pdf(d2, loc=0.0, scale=1.0)

    pdf_values = pdf_values.reshape(grid_x.shape)
    return pdf_values


def normal_logpdf_contributions_on_grid(x, y, grid_x, grid_y, sigma):
    return normal_pdf_contributions_on_grid(x, y, grid_x, grid_y, sigma, want_log=True)


def run():

    output_pickle_path = FLAGS.output_pickle_path
    grid_radius = FLAGS.grid_radius
    grid_nbr_points = FLAGS.grid_nbr_points

    dirname = os.path.dirname(output_pickle_path)
    assert os.path.exists(dirname), ("%s doesn't exist" % dirname)
    assert output_pickle_path[-4:] == ".pkl"
    assert 0.0 < grid_radius
    assert 1 <= grid_nbr_points

    spiral_noise_sigma = FLAGS.spiral_noise_sigma
    assert 1e-32 < spiral_noise_sigma

    nbr_iter = FLAGS.nbr_iter
    assert 1 <= nbr_iter

    grid_x, grid_y = np.meshgrid(np.linspace(-grid_radius, grid_radius, grid_nbr_points),
                                 np.linspace(-grid_radius, grid_radius, grid_nbr_points),
                                 sparse=False, indexing='ij')

    #for i in range(FLAGS.grid_nbr_points):
    #    for j in range(FLAGS.grid_nbr_points):
    #        # treat xv[i,j], yv[i,j]

    # Generate data points, then add the contributions of their density
    # defined by their `spiral_noise_sigma`.
    # This should have much less variance than if we did it purely by sampling.

    L_log_cumulative_weights = [] #np.zeros_like(grid_x)

    try:
        import progressbar
        # Not necessary but always nice to have when running long jobs.
        widgets = [ 'Sampling spiral: ', progressbar.Percentage(),
                    ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
                    ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=nbr_iter-1).start()
    except:
        pbar = None

    for n in range(nbr_iter):
        data = denoising_autoencoder.spiral.sample(N=1,
                                                    noise_sigma=0.0,
                                                    want_even_mass_spread=True,
                                                    angle_restriction=1.0)
        assert data.shape == (1, 2)
        (x, y) = (data[0, 0], data[0, 1])

        # Old way.
        # pdf_values = normal_pdf_contributions_on_grid(x, y, grid_x, grid_y, spiral_noise_sigma)
        # A_cumulative_weights = A_cumulative_weights + pdf_values

        # Do the log instead to stabilize numerically.
        log_pdf_values = normal_logpdf_contributions_on_grid(x, y, grid_x, grid_y, spiral_noise_sigma)
        L_log_cumulative_weights.append(log_pdf_values)

        if pbar is not None:
            pbar.update(n)
        else:
            print("%d / %d" % (n, nbr_iter))
    print("")

    A = np.concatenate([np.expand_dims(e, axis=2) for e in L_log_cumulative_weights], axis=2)
    A_log_cumulative_weights = logsumexp(A, axis=2)

    # store some extra keys to avoid certain mistakes later when aggregating
    # those files
    results = {"A_log_cumulative_weights" : A_log_cumulative_weights,
                "grid_x" : grid_x, "grid_y" : grid_y,
                # then store also the parameters
                "grid_nbr_points" : grid_nbr_points,
                "grid_radius" : grid_radius,
                "spiral_noise_sigma" : spiral_noise_sigma,
                "nbr_iter" : nbr_iter,
                }
    print("Will create %s." % output_pickle_path)
    with tf.gfile.GFile(output_pickle_path, mode='wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Wrote %s." % output_pickle_path)






if __name__ == "__main__":
    run()
