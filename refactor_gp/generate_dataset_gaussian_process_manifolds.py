#!/bin/env python

import numpy as np
import sys, os

import gaussian_process

def generate_pinched_twin_bumps_quasi_mirrored_gp(d):

    # doesn't make much sense unless we ask for
    # a dimension higher than 2
    assert d > 2

    kernel_stddev = 0.5
    def kernel(x1,x2):
        N = x1.shape[0]
        M = x2.shape[0]
        x1 = np.tile(x1.reshape((-1,1)), (1,M))
        x2 = np.tile(x2.reshape((1,-1)), (N,1))
        return np.exp(-0.5*((x1-x2)/kernel_stddev)**2)

    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    x_star = np.linspace(0.0, 1.0, d)
    obs_noise_stddev = 0.1

    # top arc
    y = np.array([0.0, 1.0,  0.5, 0.25, 0.0])
    top_arc = gaussian_process.sample_trajectory_1D(x, y, kernel, x_star, obs_noise_stddev)["samples"]

    # bottom arc (some kind of projection through x=0.5)
    y = np.array([0.0, -0.25, -0.5, -1.0,  0.0])
    bottom_arc = gaussian_process.sample_trajectory_1D(x, y, kernel, x_star, obs_noise_stddev)["samples"]

    # might want to return more useful stuff
    return (x_star, top_arc, bottom_arc)


def main(argv):

    (x_star, top_arc, bottom_arc) = generate_pinched_twin_bumps_quasi_mirrored_gp(50)

    import matplotlib
    matplotlib.use('Agg')
    import pylab
    import matplotlib.pyplot as plt

    pylab.hold(True)
    pylab.plot(x_star, top_arc)
    pylab.plot(x_star, bottom_arc)

    pylab.draw()
    pylab.savefig("/u/alaingui/umontreal/tmp/top_bottom_arcs.png", dpi=100)
    pylab.close()


if __name__ == "__main__":
    main(sys.argv)
