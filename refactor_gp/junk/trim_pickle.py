#!/bin/env python

import numpy as np
import os, sys, time

def usage():
    print "Usage."

def main(argv):
    """
    """

    import getopt
    import cPickle

    try:
        opts, args = getopt.getopt(argv[1:], "hv", ["pickle_file=", "N="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    pickle_file = None
    N = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--pickle_file"):
            pickle_file = a
        elif o in ("--N"):
            N = int(a)

    assert pickle_file is not None
    assert N is not None
    assert N > 0

    assert os.path.exists(pickle_file)
    data = cPickle.load(open(pickle_file, "r"))
    assert N <= data.shape[0]

    import re
    prog = re.compile(r"(.*)\.pkl")
    
    m = prog.match(pickle_file)
    if m:
        output_filename = "%s_trim_%d.pkl" % (m.group(1), N)
        cPickle.dump(data[0:N,:], open(output_filename, "w"))
        print "Wrote %s" % (output_filename,)
    else:
        print "Error."



if __name__ == "__main__":
    main(sys.argv)


#
# python trim_pickle.py --N=100 --pickle_file=/data/lisatmp2/alaingui/mnist/yann/yann_train_H1.pkl
#
# find /data/lisatmp2/alaingui/mnist/yann -name '*.pkl' -exec python ${HOME}/umontreal/denoising_autoencoder/refactor_gp/junk/trim_pickle.py --N=100 --pickle_file={} \;
#
