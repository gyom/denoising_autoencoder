#!/bin/env python

import numpy as np
import sys, os, time

from gyom_utils import conj
import gaussian_mixture_tools


def sample_manifold_components(d, n_components, ratio_eigvals):

    # TODO : Maybe add a scaling factor in front of the
    #        sin functions to make certain dimensions be
    #        on a different scale than the others ?

    component_means = np.zeros((n_components, d))
    domain_t = np.linspace(0,1, n_components)
    f_parameters = []
    for i in range(0, d):
        freq_multiplier = np.random.uniform(-1,1)
        exponent = np.random.randint(1,3)
        f = lambda t: np.sin(np.pi*2*freq_multiplier * t)**exponent
        f_parameters.append({'freq_multiplier':freq_multiplier, 'exponent':exponent})
        #print "freq_multiplier = %f, exponent = %f" % (freq_multiplier, exponent)
        #r = np.random.randint(0, 2)
        #if r == 0:
        #    f = lambda t: np.sin(np.pi*2*freq_multiplier * t)**exponent
        #else:
        #    f = lambda t: np.cos(np.pi*2*freq_multiplier * t)**exponent
        component_means[:,i] = f(domain_t)
        

    covariance_matrices = gaussian_mixture_tools.generate_collection_covariance_matrices(component_means, ratio_eigvals)

    return (component_means, covariance_matrices, f_parameters)



def usage():
    print "-- python generate_dataset_gaussian_mixture_manifold.py --d=50 --n_train=10000 --n_test=10000 --ratio_eigvals=0.1 --n_components=25 --output_dir=/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/0001"
    print ""

def main(argv):
    """
       n_train
       n_test
       d is the dimension of the samples. Should be higher than 2 and preferable 10 or more.
       mixing_prop controls how much of the vector v_t we mix in with the proposal for v_{t+1}
       leading_eigenvalue
       n_components
       output_dir is the directory in which we'll write the results
    """

    import getopt
    import cPickle

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["d=", "n_train=", "n_test=", "ratio_eigvals=", "n_components=", "output_dir="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    n_train = None
    n_test = None
    d = None
    ratio_eigvals = 1.0
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
        elif o in ("--ratio_eigvals"):
            ratio_eigvals = float(a)
        elif o in ("--n_components"):
            n_components = int(a)
        elif o in ("--output_dir"):
            output_dir = a
        else:
            assert False, "unhandled option"
 
    assert n_train
    assert n_test
    assert d
    assert n_components
    assert output_dir

    start_time = time.time()

    (component_means, component_covariances, f_parameters) = sample_manifold_components(d, n_components, ratio_eigvals)
    assert component_means != None
    assert component_covariances != None

    (samples, component_indices) = gaussian_mixture_tools.sample_from_mixture(component_means, component_covariances, n_train + n_test)
    end_time = time.time()
    computational_cost_in_seconds = int(end_time - start_time)
    print "Sampling took %d seconds." % computational_cost_in_seconds

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print "Creating directory %s" % output_dir,

    extra_props = {'component_means':component_means,
                   'component_covariances':component_covariances,
                   #'n_train':n_train,
                   #'n_test':n_test,
                   'd':d,
                   'ratio_eigvals':ratio_eigvals,
                   'n_components':n_components,
                   'f_parameters':f_parameters,
                   'computational_cost_in_seconds':computational_cost_in_seconds}


    train_samples = samples[0:n_train,:]
    train_component_indices = component_indices[0:n_train]
    train_samples_filename = os.path.join(output_dir, "train_samples.pkl")
    train_samples_extra_filename = os.path.join(output_dir, "train_samples_extra.pkl")

    cPickle.dump(train_samples, open(train_samples_filename, "w"))
    cPickle.dump(conj(conj(extra_props,
                           ('n', n_train)),
                      ('component_indices', train_component_indices)),
                 open(train_samples_extra_filename, "w"))
    print "wrote " + train_samples_filename
    print "wrote " + train_samples_extra_filename


    test_samples = samples[n_train:(n_train + n_test),:]
    test_component_indices= component_indices[n_train:(n_train + n_test)]
    test_samples_filename  = os.path.join(output_dir, "test_samples.pkl")
    test_samples_extra_filename  = os.path.join(output_dir, "test_samples_extra.pkl")

    cPickle.dump(test_samples, open(test_samples_filename, "w"))
    cPickle.dump(conj(conj(extra_props,
                           ('n', n_test)),
                      ('component_indices', test_component_indices)),
                 open(test_samples_extra_filename, "w"))
    print "wrote " + test_samples_filename
    print "wrote " + test_samples_extra_filename


    for i in range(0,d-1):
        output_image_file = os.path.join(output_dir,"overview_dimensions_%d_and_%d.png" % (i,i+1))
        if samples.shape[0] > 500:
            plot_the_overview(samples[0:500,:], i, i+1, output_image_file)
        else:
            plot_the_overview(samples, i, i+1, output_image_file)
        print "wrote " + output_image_file


    for i in range(0,d-1):
        output_image_file = os.path.join(output_dir,"component_means_%d_and_%d.png" % (i,i+1))
        plot_the_overview(component_means, i, i+1, output_image_file)
        print "wrote " + output_image_file


def plot_the_overview(samples, i, j,  output_image_file):

    import matplotlib
    # This has already been specified in .scitools.cfg
    # so we don't need to explicitly pick 'Agg'.
    # matplotlib.use('Agg')
    import pylab
    import matplotlib.pyplot as plt

    pylab.hold(True)
    pylab.scatter(samples[:,i], samples[:,j])
    pylab.draw()
    pylab.savefig(output_image_file, dpi=150)
    pylab.close()



if __name__ == "__main__":
    main(sys.argv)