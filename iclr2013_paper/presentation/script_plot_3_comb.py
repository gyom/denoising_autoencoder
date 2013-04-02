#!/bin/env python

import os, sys

import infinite_capacity

import matplotlib
import pylab
import matplotlib.pyplot as plt
import numpy as np

normalizing_constant = None

def f(X):
    return np.exp(-E(X))

def E(X):
    global normalizing_constant
    if normalizing_constant is None:
        domain = np.linspace(-4, 4, 1000)
        delta = domain[1] - domain[0]
        normalizing_constant = 1
        normalizing_constant = delta*np.exp(-E(domain)).sum()
        print "Set normalizing_constant to %f" % normalizing_constant

    #return 3*(X+1)**2 + 2*(X-0)**2 + (X-1)**2 - np.log( normalizing_constant )
    a = 10
    T = 1000.0
    return 1/T*(a*(X+1)**2 + 1)*(2*a*(X-0)**2 + 1)*(3*a*(X-1)**2 + 1) - np.log( normalizing_constant )


def grad_E(X):
    h = 1e-8
    return (E(X+h) - E(X))/h



def main():

    N = 1000
    Na = N*0.3/1.5
    Nb = N - Na
    print "(N, Na, Nb) = (%d, %d, %d)" % (N, Na, Nb)
    #domain = np.linspace(-1.2, 1.2, 1000)
    domain = np.linspace(-1.5, 1.5, 1000)
    output_dir = "/u/alaingui/Dropbox/umontreal/denoising_autoencoder/iclr2013_paper/presentation"

    # ground truth for all three plots
    legend_names = {"pdf":r"$p(x)$", "E":r"$E(x)$", "grad_E":r"$\nabla E(x)$"}
    colors = {"pdf":"#f62c9e", "E":"#0e7410", "grad_E":"#0e7410"}
    for (name, func) in [("pdf", f), ("E", E), ("grad_E", grad_E)]:
        pylab.hold(True)
        pa = pylab.plot(domain, func(domain), linestyle="-", color=colors[name], linewidth="4")
        if name == "grad_E":
            pylab.plot(domain, np.zeros(domain.shape), linestyle="--", color="#000000", linewidth="2")

        pylab.legend([pa], [legend_names[name]],loc=4)

        pylab.draw()
        output_file = os.path.join(output_dir, "%s.png" % (name,) )
        pylab.savefig(output_file, dpi=200)
        print "Wrote %s" % (output_file,)
        pylab.close()


    # combined graphs CAE DAE for grad_E
    #noise_stddevs = [1.0, 0.7, 0.1, 0.07, 0.01]
    noise_stddevs = np.exp(np.linspace(0, -4.6, 10))
    for noise_stddev in noise_stddevs:
        r_cae = infinite_capacity.fit_cae_1D(domain, f(domain), noise_stddev)
        r_dae = infinite_capacity.fit_dae_1D(domain, f(domain), noise_stddev)

        pylab.hold(True)
        pa = pylab.plot(domain[Na:Nb], grad_E(domain[Na:Nb]), linestyle="-", color="#0e7410", linewidth="4", label=r"$\nabla E(x)$")
        pb = pylab.plot(domain[Na:Nb], -(r_cae[Na:Nb] - domain[Na:Nb])/noise_stddev**2, linestyle="-", color="#f9a21d", linewidth="4", label=r"CAE $\left(r(x)-x\right)/\sigma^2$")
        pc = pylab.plot(domain[Na:Nb], -(r_dae[Na:Nb] - domain[Na:Nb])/noise_stddev**2, linestyle="-", color="#411ced", linewidth="4", label=r"DAE $\left(r(x)-x\right)/\sigma^2$")
        pylab.legend([pa,pb,pc], [r"$\nabla E(x)$", r"CAE $\left(r(x)-x\right)/\sigma^2$", r"DAE $\left(r(x)-x\right)/\sigma^2$"], loc=4)
        pylab.plot(domain, np.zeros(domain.shape), linestyle="--", color="#000000", linewidth="2")
        pylab.text(-1.0, 2.0, r"$\sigma = %0.2f$" % noise_stddev, fontsize=30)
        pylab.draw()
        output_file = os.path.join(output_dir, "combined_grad_E_noise_stddev_%0.3f.png" % noise_stddev)
        pylab.savefig(output_file, dpi=200)
        print "Wrote %s" % (output_file,)
        pylab.close()

    # Now we'll be printing the reconstructed p(x),
    # but it's not really worthwhile.
    # Is it ?


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



