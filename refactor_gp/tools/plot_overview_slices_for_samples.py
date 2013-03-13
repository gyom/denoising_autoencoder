#!/bin/env python

import numpy as np
import sys, os

import matplotlib
# This has already been specified in .scitools.cfg
# so we don't need to explicitly pick 'Agg'.
# matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt

def usage():
    print "Usage : python plot_overview_slices_for_samples.py --pickled_samples_file=/u/alaingui/Documents/tmp/05_000047_langevin_grad_E/mcmc_samples.pkl"

def main(argv):
    """
    pickled_samples_file (required)
    output_dir (optional)
    """

    import getopt
    import cPickle

    print argv

    try:
        opts, args = getopt.getopt(argv[1:], "hv", ["pickled_samples_file=", "output_dir="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    pickled_samples_file = None
    # plot 500 points
    N = 500
    output_dir = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--pickled_samples_file"):
            pickled_samples_file = a
        elif o in ("--N"):
            N = int(a)
        elif o in ("--output_dir"):
            output_dir = a
        else:
            assert False, "unhandled option"
 
    assert pickled_samples_file
    assert os.path.exists(pickled_samples_file)
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(pickled_samples_file), "overview_slices")
    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)

    samples = cPickle.load(open(pickled_samples_file,"r"))
    (M,d) = samples.shape
    N = min(M, N)
    ind = np.arange(M)
    np.random.shuffle(ind)
    subsamples = samples[ind[0:N]]
    print "subsamples.shape is %s" % (str(subsamples.shape),)

    for i in range(0,d-1):
        output_image_file = os.path.join(output_dir,"overview_dimensions_%d_and_%d.png" % (i,i+1))
        print "Will attempt to generate %s" % (output_image_file,)
        plot_the_overview(subsamples, i, i+1, output_image_file)
        print "Wrote %s." % ( output_image_file, )


def plot_the_overview(samples, i, j, output_image_file, dpi=150):

    pylab.hold(True)
    #print samples[:,i].shape
    #print samples[:,j].shape
    pylab.scatter(samples[:,i], samples[:,j])
    pylab.draw()
    pylab.savefig(output_image_file, dpi=dpi)
    pylab.close()



if __name__ == "__main__":
    main(sys.argv)