#!/usr/bin/env python

import dae
#dae = reload(dae)
mydae = dae.DAE(n_inputs=2,
                n_hiddens=80)

## ----------------------
## Get the training data.
## ----------------------

import debian_spiral
import numpy as np

n_spiral_samples = 5000
spiral_samples_noise_stddev = 0.0001
angle_restriction = 0.4
data = debian_spiral.sample(n_spiral_samples, spiral_samples_noise_stddev,
                            want_sorted_data = False, angle_restriction = angle_restriction)


## -----------------------------------
## Fit the model to the training data.
## -----------------------------------

batch_size = 50
n_epochs = 4000
train_noise_stddev = 0.01

if True:
    import dae_train_gradient_descent
    learning_rate = 1.0e-3
    dae_train_gradient_descent.fit(mydae,
                                   data,
                                   batch_size, n_epochs,
                                   train_noise_stddev, learning_rate,
                                   verbose=True)
else:
    import dae_train_hmc
    L = 10
    epsilon = 0.1
    dae_train_hmc.fit(mydae,
                      data,
                      batch_size, n_epochs,
                      train_noise_stddev, L, epsilon,
                      verbose=True)

mydae.set_params_to_best_noisy()
# mydae.set_params_to_best_noiseless()


## --------------------------------------
## Produce a report of the trained model.
## --------------------------------------

import os

# create a new directory to host the result files of this experiment
output_directory = '/u/alaingui/umontreal/denoising_autoencoder/plots/experiment_%0.6d' % int(np.random.random() * 1.0e6)
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
interesting_index = numpy.where( mydae.logging['noiseless']['mean_abs_loss'] < 0.1 * mydae.logging['noiseless']['mean_abs_loss'][-1] )[0][0]

plot_training_loss_history(mydae, os.path.join(output_directory, 'absolute_losses_start_to_interesting_index.png'), interesting_index)
plot_training_loss_history(mydae, os.path.join(output_directory, 'absolute_losses_start_to_end.png'), -1)


################################
##                            ##
## spiral reconstruction grid ##
##                            ##
################################

def plot_grid_reconstruction_grid(mydae, outputfile, plotgrid_N_buckets = 30, window_width = 0.3):

    (plotgrid_X, plotgrid_Y) = np.meshgrid(np.arange(- window_width,
                                                     window_width,
                                                     2 * window_width / plotgrid_N_buckets),
                                           np.arange(- window_width,
                                                     window_width,
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
    pylab.scatter(data[:,0], data[:,1], c='#f9a21d')
    pylab.hold(True)
    arrows_scaling = 1.0
    pylab.quiver(plotgrid[:,0],
                 plotgrid[:,1],
                 arrows_scaling * (grid_pred[:,0] - plotgrid[:,0]),
                 arrows_scaling * (grid_pred[:,1] - plotgrid[:,1]))
    pylab.draw()
    # pylab.axis([-0.6, 0.6, -0.6, 0.6])
    # pylab.axis([-0.3, 0.3, -0.3, 0.3])
    pylab.axis([-window_width*1.0, window_width*1.0, -window_width*1.0, window_width*1.0])
    pylab.savefig(outputfile, dpi=300)
    pylab.close()

    return grid_error


plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'spiral_reconstruction_grid_full.png'),
                              plotgrid_N_buckets = 30,
                              window_width = 1.0)

plot_grid_reconstruction_grid(mydae, os.path.join(output_directory, 'spiral_reconstruction_grid_zoomed_center.png'),
                              plotgrid_N_buckets = 30,
                              window_width = 0.3)


###################################
##                               ##
## html file for showing results ##
##                               ##
###################################


html_file_path = os.path.join(output_directory, 'results.html')
f = open(html_file_path, "w")

hyperparams_contents = """
<p>nbr visible units : %d</p>
<p>nbr hidden  units : %d</p>

<p>batch size : %d</p>
<p>epochs : %d</p>

<p>learning rate  : %0.6f</p>
<p>training noise : %0.6f</p>

<p>dataset points : %d</p>
<p>dataset noise : %0.6f</p>
<p>angle restriction : %0.6f</p>
""" % (mydae.n_inputs,
       mydae.n_hiddens,
       batch_size,
       n_epochs,
       learning_rate,
       train_noise_stddev,
       n_spiral_samples,
       spiral_samples_noise_stddev,
       angle_restriction)

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
    <img src='spiral_reconstruction_grid_full.png' width='600px'/>
    <img src='spiral_reconstruction_grid_zoomed_center.png' width='600px'/>
</div>

</body>
</html>""" % (hyperparams_contents,
              params_contents)

f.write(contents)
f.close()


