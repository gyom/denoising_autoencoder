

import numpy as np
import cPickle
import os
import sys

import subprocess


def usage():
    print "Error."

def main(argv):
    """

    """

    import getopt
    import cPickle

    try:
        #opts, args = getopt.getopt(sys.argv[1:], "hv", ["input_x=", "input_h0=", "input_h1=", "input_rh0=", "output_h0=", "output_h1=", "output_rh0=", "output_rx="])
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["root_dir=", "force"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    verbose = False

    force = False
    root_dir = None
    output_digits_png = None

    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--root_dir"):
            root_dir = a
        elif o in ("--force"):
            force = True

    assert os.path.exists(root_dir)

    s = subprocess.check_output("find %s -name mcmc_samples.pkl" % (root_dir,), shell=True)
    ready_directories = [os.path.dirname(e) for e in s.split("\n") if len(e) > 0]

    for dir in ready_directories:
        #print "Looking into %s" % (dir,)
        input_h1 = os.path.join(dir, "mcmc_samples.pkl")
        output_rx = os.path.join(dir, "mcmc_samples_rx.pkl")
        output_digits_png = os.path.join(dir, "mcmc_samples_digits.png")

        if (os.path.exists(output_rx) or os.path.exists(output_digits_png)) and (not force):
            #print "Result files already exist so we go to the next element."
            continue

        cmd = "python /u/alaingui/umontreal/denoising_autoencoder/nips2013_top_layer/process_yann_mnist_on_pickle.py --input_h1='%s' --output_rx='%s' --output_digits_png='%s'" % (input_h1, output_rx, output_digits_png)
        print cmd
        res = subprocess.check_output(cmd, shell=True)
        print res

if __name__ == "__main__":
    main(sys.argv)
