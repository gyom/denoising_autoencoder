#!/bin/env python

import numpy as np
import sys, os

import gaussian_process

from . import gyom_utils
from gyom_utils import conj

def generate_pinched_twin_bumps_quasi_mirrored_gp(d):

    # doesn't make much sense unless we ask for
    # a dimension higher than 2
    assert d > 2

    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    x_star = np.linspace(0.0, 1.0, d)
    obs_noise_stddev = 0.00001
    kernel_stddev = 10.0

    def kernel(x1,x2):
        return gaussian_process.square_distance_kernel_1D(x1,x2,kernel_stddev)

    # top arc
    top_y = np.array([0.0, 1.0,  0.5, 0.25, 0.0])
    top_arc = gaussian_process.sample_trajectory_1D(x, top_y, kernel, x_star, obs_noise_stddev)["samples"]

    # bottom arc (some kind of projection through x=0.5)
    bottom_y = np.array([0.0, -0.25, -0.5, -1.0,  0.0])
    bottom_arc = gaussian_process.sample_trajectory_1D(x, bottom_y, kernel, x_star, obs_noise_stddev)["samples"]

    # maybe we would want to funnel through the loglikelihood
    # as well and not just return the samples
    #def arc_sampler(x_star, arc_points, n_samples, sampler_kernel_stddev, obs_noise_stddev):
    #    return gaussian_process.sample_trajectory_1D(x_star, arc_points,
    #                                                 lambda x1,x2: gaussian_process.square_distance_kernel_1D(x1,x2,sampler_kernel_stddev),
    #                                                 x_star, obs_noise_stddev, n_samples)["samples"]
    #
    #def top_arc_sampler(n_samples, sampler_kernel_stddev, obs_noise_stddev):
    #    return arc_sampler(x_star, top_arc, n_samples, sampler_kernel_stddev, obs_noise_stddev)
    #
    #def bottom_arc_sampler(n_samples, sampler_kernel_stddev, obs_noise_stddev):
    #    return arc_sampler(x_star, bottom_arc, n_samples, sampler_kernel_stddev, obs_noise_stddev)

    # might want to return more useful stuff
    return (x_star, top_arc, bottom_arc, obs_noise_stddev, kernel_stddev)



def usage():
    print "-- python generate_dataset_gaussian_process_manifolds.py --d=50 --n_train=1000 --n_test=1000 --output_dir=/u/alaingui/Documents/tmp/gp_dataset_294343"

def main(argv):
    """
       n_train
       n_test
       d is the dimension of the samples. Should be higher than 2 and preferable 10 or more..
       output_dir is the directory in which we'll write the results
    """

    import getopt
    import cPickle

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["n_train=", "n_test=", "d=", "output_dir="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    n_train = None
    n_test = None
    d = None
    output_dir = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--n_train"):
            n_train = int(a)
        elif o in ("--n_test"):
            n_test = int(a)
        elif o in ("--d"):
            d = int(a)
        elif o in ("--output_dir"):
            output_dir = a
        else:
            assert False, "unhandled option"
 
    assert n_train
    assert n_test
    assert d
    assert output_dir

    # These points are used to define the arcs from which the samples are drawn.
    base_number_of_points = 8
    (base_x, top_arc, bottom_arc, 
     twin_bumps_obs_noise_stddev,
     twin_bumps_kernel_stddev) = generate_pinched_twin_bumps_quasi_mirrored_gp(base_number_of_points)

    # These values are hardcoded to yield something that looks good.
    # We're not really interested in varying those with parameters.
    kernel_stddev = 0.15
    obs_noise_stddev = 0.2
    def kernel(x1,x2):
        return gaussian_process.square_distance_kernel_1D(x1,x2,kernel_stddev)

    samples_x = np.linspace(0.0, 1.0, d)

    N = n_train + n_test
    samples = np.zeros((N, d))

    # Track from which arc you get the sample.
    cluster_index = np.array(np.random.uniform(0,1,size=N) < 0.5, dtype=int)

    # Not the most efficient way to do this because it
    # recomputes certain matrices instead of caching them,
    # but that's not important.
    for n in range(N):
        R = gaussian_process.sample_trajectory_1D(
            base_x,
            top_arc * cluster_index[n] + bottom_arc * (1 - cluster_index[n]),
            kernel, 
            samples_x,
            obs_noise_stddev)
        samples[n,:] = R['samples']
        # always the same f_star_mean and f_star_cov
        f_star_mean = R['f_star_mean']
        f_star_cov  = R['f_star_cov']


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print "Creating directory %s" % output_dir,

    extra_props = {'base_x':base_x,
                   'top_arc':top_arc,
                   'bottom_arc':bottom_arc,
                   'samples_x':samples_x,
                   'f_star_mean':f_star_mean,
                   'f_star_cov':f_star_cov,
                   'n':None,
                   'd':d,
                   'kernel_stddev':kernel_stddev,
                   'obs_noise_stddev':obs_noise_stddev,
                   'base_number_of_points':base_number_of_points,
                   'twin_bumps_obs_noise_stddev':twin_bumps_obs_noise_stddev,
                   'twin_bumps_kernel_stddev':twin_bumps_kernel_stddev}

    #print type(extra_props)
    #print type(conj(extra_props, ('n', n_train)))
    #print type(dict(extra_props.items() + [('lupi',"chien")]))
    #quit()

    train_samples = samples[0:n_train,:]
    train_cluster_index = cluster_index[0:n_train]
    train_samples_filename = os.path.join(output_dir, "train_samples.pkl")
    train_samples_extra_filename = os.path.join(output_dir, "train_samples_extra.pkl")

    cPickle.dump(train_samples, open(train_samples_filename, "w"))
    cPickle.dump(conj(conj(extra_props,
                           ('n', n_train)),
                      ('cluster_indices', train_cluster_index)),
                 open(train_samples_extra_filename, "w"))
    print "wrote " + train_samples_filename
    print "wrote " + train_samples_extra_filename


    test_samples = samples[n_train:(n_train + n_test),:]
    test_cluster_index = cluster_index[n_train:(n_train + n_test)]
    test_samples_filename  = os.path.join(output_dir, "test_samples.pkl")
    test_samples_extra_filename  = os.path.join(output_dir, "test_samples_extra.pkl")

    cPickle.dump(test_samples, open(test_samples_filename, "w"))
    cPickle.dump(conj(conj(extra_props,
                           ('n', n_test)),
                      ('cluster_indices', test_cluster_index)),
                 open(test_samples_extra_filename, "w"))
    print "wrote " + test_samples_filename
    print "wrote " + test_samples_extra_filename

    output_image_file = os.path.join(output_dir,"overview.png")
    plot_the_overview(base_x, top_arc, bottom_arc, samples_x, train_samples[0:10,:], train_samples[10:20,:], output_image_file)
    print "wrote " + output_image_file

    # TODO : Make sure that you store all the relevant parameters
    #        that will be required later to evaluate probabilities
    #        for the trajectories.



def plot_the_overview(base_x, top_arc, bottom_arc, samples_x, top_arc_samples, bottom_arc_samples, output_image_file):

    import matplotlib
    matplotlib.use('Agg')
    import pylab
    import matplotlib.pyplot as plt

    pylab.hold(True)

    for n in range(top_arc_samples.shape[0]):
        pylab.plot(samples_x, top_arc_samples[n,:], color="#c27ab9")

    for n in range(bottom_arc_samples.shape[0]):
        pylab.plot(samples_x, bottom_arc_samples[n,:], color="#c3cc81")

    pylab.plot(base_x, top_arc, color="#7a7fc1")
    pylab.plot(base_x, bottom_arc, color="#7a7fc1")

    pylab.draw()
    pylab.savefig(output_image_file, dpi=150)
    pylab.close()



if __name__ == "__main__":
    main(sys.argv)