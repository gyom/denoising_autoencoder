

import numpy as np
import cPickle
import os
import sys

import nips2013_top_layer
import nips2013_top_layer.common
from nips2013_top_layer.common import encode_h0_to_h1, encode_x_to_h0, decode_h1_to_h0, decode_h0_to_x 

def usage():
    print "Error."

def main(argv):
    """

    """

    import getopt
    import cPickle

    try:
        #opts, args = getopt.getopt(sys.argv[1:], "hv", ["input_x=", "input_h0=", "input_h1=", "input_rh0=", "output_h0=", "output_h1=", "output_rh0=", "output_rx="])
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["input_h1=", "output_rx=", "output_digits_png="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    verbose = False

    input_h1 = None
    output_rx = None
    output_digits_png = None

    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--input_h1"):
            input_h1 = a
        elif o in ("--output_rx"):
            output_rx = a
        elif o in ("--output_digits_png"):
            output_digits_png = a

    assert os.path.exists(input_h1)
    assert output_rx is not None


    ##### Performing the conversion #####

    H1 = cPickle.load(open(input_h1, "r"))
    print "Read %s" % (input_h1,)
    rH0 = decode_h1_to_h0(H1)
    rX = decode_h0_to_x(rH0)
    cPickle.dump(rX, open(output_rx, "w"))
    print "Wrote %s" % (output_rx,)


    ##### Plotting the digits if desired #####

    if output_digits_png is not None:

        import refactor_gp
        import refactor_gp.yann_dauphin_utils
        from refactor_gp.yann_dauphin_utils import tile_raster_images
        I = tile_raster_images(rX, (28,28), (int(rX.shape[0]/20) + 1, 20))

        import matplotlib
        import pylab
        import matplotlib.pyplot as plt

        from PIL import Image
        im = Image.fromarray(I)
        im.save(output_digits_png)
        #im.save("/u/alaingui/umontreal/denoising_autoencoder/refactor_gp/junk/mnist.png")
        print "Wrote %s" % (output_digits_png,)


if __name__ == "__main__":
    main(sys.argv)

