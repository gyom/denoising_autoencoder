"""
Combines many output files from `generate_discretized_density.py`.
"""


import sys
sys.path.insert(0,'/current_project/src')

import os
import pickle

import numpy as np

# Using tensorflow to handle parameters. This is almost silly, but we have
# tensorflow anyways for other things so we might as well use it here too.
import tensorflow as tf

import denoising_autoencoder

from denoising_autoencoder.logsumexp import logsumexp

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "input_pickle_glob", None,
    "Input glob pattern for files to read. Use wildcards.")

flags.DEFINE_string(
    "output_pickle_path", None,
    "Pickle file where we want to store the computed density.")

FLAGS = flags.FLAGS


def run():

    input_pickle_glob = FLAGS.input_pickle_glob
    output_pickle_path = FLAGS.output_pickle_path

    L_input_path = tf.gfile.Glob(input_pickle_glob)
    assert 0 < len(L_input_path), "No input files found with glob pattern."

    print("Will process the following files :")
    print(L_input_path)
    # Good! We get the full paths.
    # ['/current_project/data/p_part_01.pkl', '/current_project/data/p_part_00.pkl']



    try:
        import progressbar
        # Not necessary but always nice to have when running long jobs.
        widgets = [ 'Sampling spiral: ', progressbar.Percentage(),
                    ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
                    ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(L_input_path)-1).start()
    except:
        pbar = None


    accumulated_results = None
    for (n, input_path) in enumerate(L_input_path):
        with tf.gfile.GFile(input_path, mode='rb') as f:
            results = pickle.load(f)

        if accumulated_results is None:
            accumulated_results = results
        else:
            assert np.max(np.abs(accumulated_results["grid_x"] - results["grid_x"])) < 1e-8
            assert np.max(np.abs(accumulated_results["grid_y"] - results["grid_y"])) < 1e-8
            assert np.max(np.abs(accumulated_results["grid_nbr_points"] - results["grid_nbr_points"])) < 1e-8
            assert np.max(np.abs(accumulated_results["grid_radius"] - results["grid_radius"])) < 1e-8
            assert np.max(np.abs(accumulated_results["spiral_noise_sigma"] - results["spiral_noise_sigma"])) < 1e-8

            accumulated_results["nbr_iter"] += results["nbr_iter"]

            A = accumulated_results["A_log_cumulative_weights"]
            B = results["A_log_cumulative_weights"]
            assert len(A.shape) == 2
            assert len(B.shape) == 2

            A_log_cumulative_weights = logsumexp(
                np.concatenate([np.expand_dims(e, axis=2) for e in [A, B]], axis=2),
                axis=2)
            assert len(A_log_cumulative_weights.shape) == 2

            # This is what accumulates over many steps.
            accumulated_results["A_log_cumulative_weights"] = A_log_cumulative_weights

        if pbar is not None:
            pbar.update(n)
        else:
            print("%d / %d" % (n, len(L_input_path)))
    print("")

    with tf.gfile.GFile(output_pickle_path, mode='wb') as f:
        pickle.dump(accumulated_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Wrote %s." % output_pickle_path)


if __name__ == "__main__":
    run()
