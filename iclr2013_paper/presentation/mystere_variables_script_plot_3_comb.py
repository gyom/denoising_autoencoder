#!/bin/env python

import numpy as np

import infinite_capacity

import matplotlib
import pylab
import matplotlib.pyplot as plt


normalizing_constant = None

def f(X):
    return X

def E(X):
    global normalizing_constant
    if normalizing_constant is None:
        domain = np.linspace(-4, 4, 1000)
        delta = domain[1] - domain[0]
        normalizing_constant = 1
        normalizing_constant = delta*np.exp(-E(domain)).sum()
        print "Set normalizing_constant to %f" % normalizing_constant

    #return 3*(X+1)**2 + 2*(X-0)**2 + (X-1)**2 - np.log( normalizing_constant )
    return ((X+1)**2 + 3)*((X-0)**2 + 2)*((X-1)**2 + 1) - np.log( normalizing_constant )


#def grad_E(X):
#    #return 6*(X+1) + 4*(X-0) + 2*(X-1)
#    pass

def p(X):
    return np.exp(-E(X))


def main():

    domain = np.linspace(-4, 4, 1000) 
    print E
    print f
    print p


    pylab.hold(True)
    pylab.plot(domain, p(domain))

    pylab.draw()
    output_file = "/u/alaingui/Dropbox/umontreal/denoising_autoencoder/iclr2013_paper/presentation/1.png"
    pylab.savefig(output_file, dpi=150)
    print "Wrote %s" % (output_file,)
    pylab.close()

    if False:
        x = np.linspace(-1, 1, 100)
        p = np.exp(-x**2/2)
        noise_stddev = 0.1

        r_cae = infinite_capacity.fit_cae_1D(x,p,noise_stddev)
        print r_cae

        r_dae = infinite_capacity.fit_dae_1D(x,p,noise_stddev)
        print r_dae

        X = np.vstack( (x, np.zeros(x.shape)) ).T
        r_dae = infinite_capacity.fit_dae(X,p,noise_stddev)
        print r_dae




if __name__ == "__main__":
    main()



