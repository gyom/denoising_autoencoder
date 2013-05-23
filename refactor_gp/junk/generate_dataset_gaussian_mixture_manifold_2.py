#!/bin/env python

import numpy as np
import sys, os, time

from gyom_utils import conj


def sample_next_component(previous_leading_eigenvector, d, leading_eigenvalue, mixing_prop=0.5):

    # Note that 'previous_leading_eigenvector' can be None if
    # the 'mixing_prop' is 0.0 .

    # This whole method produces results that are somehow approximate,
    # but should be close enough to the desired behavior for the
    # manifold generate to be nice.

    w = np.zeros((d,))
    v = np.zeros((d,d))
    # Randomize until you get linearly independent vectors
    # (happens with probabiblity 1, but technically it *could* be a problem).
    want_retry = True
    while np.any(w == 0) or (np.linalg.det(v) == 0) or want_retry:

        want_retry = False

        S = np.random.multivariate_normal( mean=np.ones((d,)), cov = np.diag(np.ones((d,))), size=(10*d,) )
        w, v = np.linalg.eig(np.random.normal( S.T.dot(S) ))

        #print w
        #print v

        if not np.all(np.real(w) == w):
            print "Not all eigenvalues are real. Generating seed sample again."
            want_retry = True
            continue

        if not np.all(np.real(v) == v):
            print "Not all eigenvectors are real. Generating seed sample again."
            want_retry = True
            continue

        w = np.ones((d,))
        w[0] = leading_eigenvalue
        if mixing_prop > 0.0:
            print "previous_leading_eigenvector"
            print previous_leading_eigenvector
            print "current suggestion for leading eigenvector"
            print v[:,0]
            print
            v[:,0] = previous_leading_eigenvector * (1.0 - mixing_prop) + mixing_prop * v[:,0]

        # Now we sorta spoiled our collection of vectors because
        # they're no longer orthonormal. However, we still have
        # that they span the whole space and we get all the
        # desired properties from the fact that we generate
        # a matrix from something of the form V^-1 D V.

        covariance_matrix = np.linalg.inv(v).dot(np.diag(w)).dot(v)

        w,v = np.linalg.eig(covariance_matrix)
        assert np.all(w > 0.0)

        if not np.all(np.real(w) == w):
            print "Not all eigenvalues are real. Generating seed sample again. _"
            want_retry = True
            continue

        if not np.all(np.real(v) == v):
            print "Not all eigenvectors are real. Generating seed sample again. _"
            want_retry = True
            continue

        i = np.argmax(w)
        if np.abs(w[i] -  leading_eigenvalue) > 0.001:
            print "Leading eigenvalue %f should be close to %f, but it's not. Generating again." % (w[i], leading_eigenvalue)
            want_retry = True
            continue

    # To enforce some kind of consistency, we'll orient the
    # eigenvector in a direction that makes it have the highest
    # number of "positive" components.
    # Otherwise I'm afraid we'll be going back and forth
    # in consecutive movements.
    if np.count_nonzero(v[:,i] > 0) >  np.count_nonzero(v[:,i] < 0):
        return (v[:,i], covariance_matrix)
    else:
        return (-v[:,i], covariance_matrix)



def sample_manifold_components(d, n_components, leading_eigenvalue, mixing_prop):

    components = np.zeros((n_components, d))
    covariance_matrices = np.zeros((n_components, d, d))

    components[0,:] = np.zeros((d,))
    
    (v, covmat) = sample_next_component(None, d, leading_eigenvalue, 0.0)
    covariance_matrices[0,:,:] = covmat

    components[1,:] = leading_eigenvalue * v
    for n in range(1, n_components):
        # When entering the loop,
        #   components[n,:] is set
        #   covariance_matrices[n,:] is missing
        #
        # When leaving the loop,
        #   covariance_matrices[n,:] is set
        #   components[n+1,:] is set

        (v, covmat) = sample_next_component((components[n,:] - components[n-1,:])/leading_eigenvalue,
                                            d, leading_eigenvalue, mixing_prop)
        covariance_matrices[n,:] = covmat
        if n+1 < n_components:
            components[n+1,:] = components[n,:] + leading_eigenvalue * v

    return (components, covariance_matrices)


def sample_from_mixture(component_means, component_covariances, n_samples):

    (n_components, d) = component_means.shape
    (n_components1, d1, d2) = component_covariances.shape
    assert n_components == n_components1
    assert d == d1
    assert d == d2

    samples = np.zeros((n_samples, d))
    component_indices = np.zeros((n_samples,))
    for k in np.arange(n_samples):
        c = np.random.randint(n_components)
        component_indices[k] = c
        samples[k,:] = np.random.multivariate_normal(mean=component_means[c,:], cov=component_covariances[c,:,:])
        
    return (samples, component_indices)




def usage():
    print "-- python generate_dataset_gaussian_mixture_manifold.py --d=50 --n_train=10000 --n_test=10000 --mixing_prop=0.5 --leading_eigenvalue=10.0 --n_components=10 --output_dir=/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture"
    print ""
    print "mixing_prop being 0.0 will get you nowhere. With 1.0 you get the most randomness for the manifold direction."

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
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["d=", "n_train=", "n_test=", "mixing_prop=", "leading_eigenvalue=", "n_components=", "output_dir="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    n_train = None
    n_test = None
    d = None
    mixing_prop = 0.5
    leading_eigenvalue = 1.0
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
        elif o in ("--mixing_prop"):
            mixing_prop = float(a)
        elif o in ("--leading_eigenvalue"):
            leading_eigenvalue = float(a)
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

    (component_means, component_covariances) = sample_manifold_components(d, n_components, leading_eigenvalue, mixing_prop)
    (samples, component_indices) = sample_from_mixture(component_means, component_covariances, n_train + n_test)

    end_time = time.time()
    computational_cost_in_seconds = int(end_time - start_time)
    print "Sampling took %d seconds." % computational_cost_in_seconds

    print component_means


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print "Creating directory %s" % output_dir,

    extra_props = {'component_means':component_means,
                   'component_covariances':component_covariances,
                   #'n_train':n_train,
                   #'n_test':n_test,
                   'd':d,
                   'leading_eigenvalue':leading_eigenvalue,
                   'mixing_prop':mixing_prop,
                   'n_components':n_components,
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
        plot_the_overview(samples, i, i+1, output_image_file)
        print "wrote " + output_image_file

    # TODO : Make sure that you store all the relevant parameters
    #        that will be required later to evaluate probabilities
    #        for the trajectories.



import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt


def plot_the_overview(samples, i, j,  output_image_file):

    pylab.hold(True)
    pylab.scatter(samples[:,i], samples[:,j])
    pylab.draw()
    pylab.savefig(output_image_file, dpi=150)
    pylab.close()



if __name__ == "__main__":
    main(sys.argv)