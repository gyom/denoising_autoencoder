
#!/usr/bin/env python

import sys
import os
import numpy as np


def make_energy(W, b, c, alpha, beta):
    if (len(b.shape) != 1 or
        len(c.shape) != 1 or
        len(W.shape) != 2 or
        b.shape[0] != W.shape[0] or
        c.shape[0] != W.shape[1] or
        type(alpha) != type(1.0) or
        type(beta) != type(1.0)):
        error("Wrong dimensions in one of W, b, c, alpha, beta.")

    def overflow_protection(A):
        # find something to do like the softplus log(1+exp(x)) that turns into x when x > 70
        B = np.abs(A) - np.log(2)
        I = np.where(np.abs(A)<50)
        B[I] = np.log(np.abs(np.sinh(A[I])))
        return B

    def E(X):
        B = overflow_protection(X.dot(W) + c)
        main_term = B.sum(axis=1).reshape((-1,)) - X.dot(b.reshape((-1,1))).reshape((-1,))
        exponential_penalty = 0.5 * (X**2).sum(axis=1).reshape((-1,))

        # It's not clear right now what is the correct way to do this.
        #
        # 's' applied to both terms
        #return s * (- main_term + exponential_penalty)
        # 's' applied only to the main term and not the exponential penalty
        return - alpha * main_term + beta * exponential_penalty)
    return E



n_hiddens = 16
n_inputs = 2

model_W = np.random.normal(size=(n_inputs, n_hiddens), scale = 10)
model_b = np.random.normal(size=(n_inputs,))
model_c = np.random.normal(size=(n_hiddens,))
model_alpha = 1.0
model_beta = 1.0

print "Got the energy model."
E_ = make_energy(model_W, model_b, model_c, model_alpha, model_beta)
E = lambda x: E_(x.reshape((1,-1)))


import metropolis_hastings_sampler
x0 = np.zeros((n_inputs,))
symmetric_proposal = lambda x: x + np.random.normal(size=x.shape, scale = 0.1)
N = 1000
(X, acceptance_ratio) = metropolis_hastings_sampler.run_chain_with_energy(E, x0, symmetric_proposal, N, thinning_factor = 100, burn_in = 10000)

print "Got the samples. Acceptance ratio was %f" % acceptance_ratio




if True:
    import matplotlib
    matplotlib.use('Agg')
    import pylab

    pylab.scatter(X[:,0], X[:,1])
    pylab.draw()
    pylab.savefig("/u/alaingui/umontreal/tmp/demo2.png", dpi=300)
    pylab.close()    

    quit()




#N = 1000
#X = np.zeros((N,2))
#X[:,0] = np.linspace(-1,1,N)
#X[:,1] = np.sin( 2* np.pi * np.linspace(-1,1,N) )

print X.shape

from dae_untied_weights import DAE_untied_weights

mydae = DAE_untied_weights(n_inputs = 2,
                           n_hiddens = 32,
                           act_func = ['tanh', 'tanh'],
                           want_plus_x = False)


mydae.fit_with_decreasing_noise(X,
                                [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001],
                                {'method' : 'fmin_bfgs',
                                 'maxiter' : 5000})


clean_data = X

import matplotlib
matplotlib.use('Agg')
import pylab

def plot_grid_reconstruction_grid(mydae, outputfile, plotgrid_N_buckets = 30, window_width = 1.0, center = (0.0,0.0)):

    (plotgrid_X, plotgrid_Y) = np.meshgrid(np.arange(center[0] - window_width,
                                                     center[0] + window_width,
                                                     2 * window_width / plotgrid_N_buckets),
                                           np.arange(center[1] - window_width,
                                                     center[1] + window_width,
                                                     2 * window_width / plotgrid_N_buckets))
    plotgrid = np.vstack([np.hstack(plotgrid_X), np.hstack(plotgrid_Y)]).T

    print "Making predictions for the grid."

    grid_pred = mydae.encode_decode(plotgrid)
    grid_error = np.sqrt(((grid_pred - plotgrid)**2).sum(axis=1)).mean()
    print "grid_error = %0.6f (not necessarily a relevant information)" % grid_error


    print "Generating plot."

    #import matplotlib.pyplot as plt
    #plt.hexbin(clean_data[:,0], clean_data[:,1], bins='log', cmap=plt.cm.YlOrRd_r)
    #plt.hexbin(clean_data[:,0], clean_data[:,1], cmap=plt.cm.YlOrRd_r)
    pylab.scatter(clean_data[:,0], clean_data[:,1])

    pylab.hold(True)
    arrows_scaling = 1.0
    pylab.quiver(plotgrid[:,0],
                 plotgrid[:,1],
                 arrows_scaling * (grid_pred[:,0] - plotgrid[:,0]),
                 arrows_scaling * (grid_pred[:,1] - plotgrid[:,1]))
    pylab.draw()
    pylab.axis([center[0] - window_width*1.0, center[0] + window_width*1.0,
                center[1] - window_width*1.0, center[1] + window_width*1.0])
    pylab.savefig(outputfile, dpi=300)
    pylab.close()

    return grid_error


output_directory = '/u/alaingui/Documents/tmp'
plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'demo_2_tight.png'),
                              plotgrid_N_buckets = 50,
                              window_width = 1.0)
plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'demo_2_zoomedout.png'),
                              plotgrid_N_buckets = 50,
                              window_width = 2.0)
