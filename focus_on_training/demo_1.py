
#!/usr/bin/env python

import sys
import os
import numpy as np

N = 1000
X = np.zeros((N,2))
X[:,0] = np.linspace(-1,1,N)
X[:,1] = np.sin( 2* np.pi * np.linspace(-1,1,N) )

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
plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'demo_1_tight.png'),
                              plotgrid_N_buckets = 50,
                              window_width = 1.0)
plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'demo_1_zoomedout.png'),
                              plotgrid_N_buckets = 50,
                              window_width = 2.0)
