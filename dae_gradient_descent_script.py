#!/usr/bin/env python

import dae
#dae = reload(dae)
mydae = dae.DAE(n_inputs=2,
                n_hiddens=800)

## ----------------------
## Get the training data.
## ----------------------

import debian_spiral
import numpy as np

n_spiral_samples = 1000
spiral_samples_noise_stddev = 0.01
(X,Y) = debian_spiral.sample(n_spiral_samples, spiral_samples_noise_stddev)
data = np.vstack((X,Y)).T


## -----------------------------------
## Fit the model to the training data.
## -----------------------------------

import dae_train_gradient_descent

batch_size = 25
n_epochs = 1000
train_noise_stddev = 0.1
learning_rate = 1.0e-3
dae_train_gradient_descent.fit(mydae,
                               data,
                               batch_size, n_epochs,
                               train_noise_stddev, learning_rate,
                               verbose=True)

mydae.set_params_to_best_noisy()

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


pylab.hold(True)

p1, = pylab.plot(mydae.logging['noisy']['mean_abs_loss'], label='noisy', c='#f9761d', linewidth = 2)
for s in [-1.0, 1.0]:
    pylab.plot(mydae.logging['noisy']['mean_abs_loss']
               + s * np.sqrt(mydae.logging['noisy']['var_abs_loss']),
               c='#f9a21d', linestyle='dashed')

p2, = pylab.plot(mydae.logging['noiseless']['mean_abs_loss'], label='noiseless', c='#9418cd', linewidth = 2)
for s in [-1.0, 1.0]:
    pylab.plot(mydae.logging['noiseless']['mean_abs_loss']
               + s * np.sqrt(mydae.logging['noiseless']['var_abs_loss']),
               c='#d91986', linestyle='dashed')

pylab.title('Absolute Losses')
pylab.legend([p1,p2], ["noisy", "noiseless"])
pylab.draw()
pylab.savefig(os.path.join(output_directory, 'absolute_losses.png'), dpi=300)
pylab.close()




################################
##                            ##
## spiral reconstruction grid ##
##                            ##
################################

plotgrid_N_buckets = 30
window_width = 1.0
(plotgrid_X, plotgrid_Y) = np.meshgrid(np.arange(- window_width,
                                                       window_width,
                                                       2 * window_width / plotgrid_N_buckets),
                                          np.arange(- window_width,
                                                       window_width,
                                                       2 * window_width / plotgrid_N_buckets))
plotgrid = np.vstack([np.hstack(plotgrid_X), np.hstack(plotgrid_Y)]).T
D = np.sqrt(plotgrid[:,0]**2 + plotgrid[:,1]**2)
plotgrid = plotgrid[D<0.7]

print plotgrid_X.shape
print plotgrid_Y.shape

print "Will keep only %d points on the plotting grid after starting from %d." % (plotgrid.shape[0], plotgrid_X.shape[0])

print "Making predictions for the grid."

grid_pred = mydae.encode_decode(plotgrid)
#grid_pred = predict(plotgrid, W, grid, (lambda X, xi: kernel(X,xi,sigma)))
grid_error = np.sqrt(((grid_pred - plotgrid)**2).sum(axis=1)).mean()
print "grid_error = %0.6f" % grid_error


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
#pylab.axis([-0.6, 0.6, -0.6, 0.6])
pylab.axis([-0.7, 0.7, -0.7, 0.7])
# pylab.axis([-window_width*1.5, window_width*1.5, -window_width*1.5, window_width*1.5])
pylab.savefig(os.path.join(output_directory, 'spiral_reconstruction_grid.png'), dpi=300)
pylab.close()



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
""" % (mydae.n_inputs,
       mydae.n_hiddens,
       batch_size,
       n_epochs,
       learning_rate,
       train_noise_stddev,
       n_spiral_samples,
       spiral_samples_noise_stddev)

params_contents = ""


graphs_contents = """
<img src='%s' width='600px'/>
<img src='%s' width='600px'/>
""" % ('absolute_losses.png', 'spiral_reconstruction_grid.png')


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
<h3>Graphs</h3>
<div class='listing'>%s</div>
</body>
</html>""" % (hyperparams_contents,
              params_contents,
              graphs_contents)

f.write(contents)
f.close()


