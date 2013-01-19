#!/usr/bin/env python


import sys

# Having an option to read some parameters as json fed
# as argument.
if len(sys.argv) > 1:
    import json
    override_params = json.loads(sys.argv[1])
    print override_params
    for k in override_params.keys():
        if not k in ['train_noise_stddev',
                     'maxiter',
                     'n_hiddens']:
            error('You probably misspelled a parameter to override.')
else:
    override_params = {}

import numpy as np
#np.random.seed(38730)

import dae
#dae = reload(dae)
n_inputs = 2

if 'n_hiddens' not in override_params.keys():
    n_hiddens = 250
else:
    n_hiddens = override_params['n_hiddens']
output_scaling_factor = 40.0

mydae = dae.DAE(n_inputs = n_inputs,
                n_hiddens = n_hiddens,
                output_scaling_factor = output_scaling_factor)

## ----------------------
## Get the training data.
## ----------------------

def sample_two_bumps(N):
    A = np.array([[1,1],[1,-1]])
    D = np.array([[40,0],[0,5]])
    S = A.dot(D.dot(np.linalg.inv(A)))

    samples = np.random.multivariate_normal([5.0, 0.0],S,N)
    ind_to_flip = np.where(np.random.uniform(0,1,(N,))<0.5)[0]
    samples[ind_to_flip,0] = -samples[ind_to_flip,0]

    return samples



n_original_samples = 10000
original_data = sample_two_bumps(n_original_samples)
if original_data.shape != (n_original_samples, 2):
    error("Wrong shape for original_data.")
replication_factor = 10
n_replicated_samples = n_original_samples * replication_factor

if 'train_noise_stddev' not in override_params.keys():
    train_noise_stddev = 1.00
else:
    train_noise_stddev = override_params['train_noise_stddev']
clean_data = np.tile(original_data, (replication_factor, 1))
np.random.shuffle(clean_data)
noisy_data = clean_data + np.random.normal(size = clean_data.shape,
                                           scale = train_noise_stddev)

if clean_data.shape != (n_replicated_samples, 2):
    error("Wrong shape for the data.")


## -----------------------------------
## Fit the model to the training data.
## -----------------------------------

# We'll use a multi-stage process later.

import dae_train_scipy_optimize
optimization_args = {'method' : 'fmin_bfgs',
                     'gtol' : 1.0e-3,
                     'maxiter' : 1000,
                     'avextol' : 1.0e-4}
if 'maxiter' in override_params.keys():
    maxiter = override_params['maxiter']

dae_train_scipy_optimize.fit(mydae,
                             X = clean_data,
                             noisy_X = noisy_data,
                             optimization_args = optimization_args,
                             verbose=True)

## --------------------------------------
## Produce a report of the trained model.
## --------------------------------------

# We have to get a random value for the name of the
# directory used to put the output results, but we
# can't rely on the currently used random seed
# or otherwise all the results will go the same
# directory.

import os
# import time
# np.random.seed(int(time.time()))

if os.getenv("DENOISING_REPO")=="":
   print "Please define DENOISING_REPO environment variable"
   quit()
output_directory = os.path.join(os.getenv("DENOISING_REPO"),
                                'denoising_autoencoder/plots_two_bumps/experiment_%0.6d' % int(np.random.random() * 1.0e6))

if not os.path.exists(output_directory):
    os.makedirs(output_directory)





import matplotlib
matplotlib.use('Agg')
import pylab

def plot_training_loss_history(mydae, outputfile, last_index = -1):
    pylab.hold(True)
    p1, = pylab.plot(mydae.logging['noisy']['mean_abs_loss'][:last_index], label='noisy', c='#f9761d', linewidth = 2)
    for s in [-1.0, 1.0]:
        pylab.plot(mydae.logging['noisy']['mean_abs_loss'][:last_index]
                   + s * np.sqrt(mydae.logging['noisy']['var_abs_loss'][:last_index]),
                   c='#f9a21d', linestyle='dashed')

        p2, = pylab.plot(mydae.logging['noiseless']['mean_abs_loss'][:last_index], label='noiseless', c='#9418cd', linewidth = 2)
        for s in [-1.0, 1.0]:
            pylab.plot(mydae.logging['noiseless']['mean_abs_loss'][:last_index]
                       + s * np.sqrt(mydae.logging['noiseless']['var_abs_loss'][:last_index]),
                       c='#d91986', linestyle='dashed')

            pylab.title('Absolute Losses')
            pylab.legend([p1,p2], ["noisy", "noiseless"])
            pylab.draw()
            pylab.savefig(outputfile, dpi=300)
            pylab.close()

# We want to pick some interesting index at which most of the learning has already taken place.
A = mydae.logging['noiseless']['mean_abs_loss'] < 4.0 * mydae.logging['noiseless']['mean_abs_loss'][-1]
if np.any(A):
    interesting_index = np.where( A )[0][0]
else:
    interesting_index = min(100, len(mydae.logging['noiseless']['mean_abs_loss']))
del A
print "interesting_index = %d" % interesting_index

plot_training_loss_history(mydae, os.path.join(output_directory, 'absolute_losses_start_to_interesting_index.png'), interesting_index)
plot_training_loss_history(mydae, os.path.join(output_directory, 'absolute_losses_start_to_end.png'), -1)


################################
##                            ##
## two_bumps reconstruction grid ##
##                            ##
################################

def plot_grid_reconstruction_grid(mydae, outputfile, plotgrid_N_buckets = 30, window_width = 0.3, center = (0.0,0.0)):

    (plotgrid_X, plotgrid_Y) = np.meshgrid(np.arange(center[0] - window_width,
                                                     center[0] + window_width,
                                                     2 * window_width / plotgrid_N_buckets),
                                           np.arange(center[1] - window_width,
                                                     center[1] + window_width,
                                                     2 * window_width / plotgrid_N_buckets))
    plotgrid = np.vstack([np.hstack(plotgrid_X), np.hstack(plotgrid_Y)]).T

    # Not sure it's worth truncating some elements now that we're
    # producing more plots.
    #    D = np.sqrt(plotgrid[:,0]**2 + plotgrid[:,1]**2)
    #    plotgrid = plotgrid[D<0.7]
    #    print plotgrid_X.shape
    #    print plotgrid_Y.shape
    #    print "Will keep only %d points on the plotting grid after starting from %d." % (plotgrid.shape[0], plotgrid_X.shape[0])

    print "Making predictions for the grid."

    grid_pred = mydae.encode_decode(plotgrid)
    grid_error = np.sqrt(((grid_pred - plotgrid)**2).sum(axis=1)).mean()
    print "grid_error = %0.6f (not necessarily a relevant information)" % grid_error


    print "Generating plot."

    # print only one point in 100
    # pylab.scatter(data[0:-1:100,0], data[0:-1:100,1], c='#f9a21d')
    #pylab.scatter(clean_data[:,0], clean_data[:,1], c='#f9a21d')
    import matplotlib.pyplot as plt
    plt.hexbin(clean_data[:,0], clean_data[:,1], bins='log', cmap=plt.cm.YlOrRd_r)
    # doesn't work for hist2d
    #from matplotlib import pyplot
    #pyplot.hist2d(clean_data[:,0], clean_data[:,1],nbins=100)

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


plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'two_bumps_reconstruction_grid_full.png'),
                              plotgrid_N_buckets = 30,
                              window_width = 1.0)

plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'two_bumps_reconstruction_grid_zoomed_center.png'),
                              plotgrid_N_buckets = 30,
                              window_width = 5.0)

plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'two_bumps_reconstruction_grid_half.png'),
                              plotgrid_N_buckets = 30,
                              window_width = 10.0)


plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'two_bumps_reconstruction_grid_zoomed_close_manifold.png'),
                              plotgrid_N_buckets = 30,
                              window_width = 50.0)


###################################
##                               ##
## html file for showing results ##
##                               ##
###################################


html_file_path = os.path.join(output_directory, 'results.html')
f = open(html_file_path, "w")

hyperparams_contents_for_method = ""

hyperparams_contents = """
<p>nbr visible units : %d</p>
<p>nbr hidden  units : %d</p>

%s

<p>training noise : %0.6f</p>

<p>original dataset points   : %d</p>
<p>replicated dataset points : %d</p>
""" % (mydae.n_inputs,
       mydae.n_hiddens,
       hyperparams_contents_for_method,
       train_noise_stddev,
       n_original_samples,
       n_replicated_samples)

params_contents = ""


contents = """
<html>
    <head>
        <style>
            div.listing {
                margin-left: 25px;
                margin-top: 5px;
                margin-botton: 5px;
            } 
        </style>
    </head>
<body>

<h3>Hyperparameters</h3>
<div class='listing'>%s</div>

<h3>Parameters</h3>
<div class='listing'>%s</div>

<h3>Training Loss</h3>
<div class='listing'>
    <img src='absolute_losses_start_to_interesting_index.png' width='600px'/>
    <img src='absolute_losses_start_to_end.png' width='600px'/>
</div>

<h3>Training Loss</h3>
<div class='listing'>
    <img src='two_bumps_reconstruction_grid_full.png' width='600px'/>
    <img src='two_bumps_reconstruction_grid_zoomed_center.png' width='600px'/>
    <img src='two_bumps_reconstruction_grid_zoomed_close_manifold.png' width='600px'/>
    <img src='two_bumps_reconstruction_grid_half.png' width='600px'/>
</div>

</body>
</html>""" % (hyperparams_contents,
              params_contents)

f.write(contents)
f.close()
print("Wrote " + html_file_path)

##################################
#
# Pickle the parameters for reuse.
#
##################################

import cPickle
pickled_results_file = os.path.join(output_directory, 'results.pkl')
f = open(pickled_results_file, "w")
cPickle.dump({'Wb' : mydae.Wb,
              'Wc' : mydae.Wc,
              'b' : mydae.b,
              'c' : mydae.c,
              'optimization_args' : optimization_args,
              'n_inputs' : n_inputs,
              'n_hiddens' : n_hiddens,
              'output_scaling_factor' : output_scaling_factor,
              'train_noise_stddev' : train_noise_stddev},
             f)
f.close()
print("Wrote " + pickled_results_file)