#!/usr/bin/env python

import dae
import debian_spiral
import numpy as np

np.random.seed(99)

#dae = reload(dae)
mydae = dae.DAE(n_inputs=1,
                n_hiddens=4)

## ----------------------
## Get the training data.
## ----------------------

# Three training points, repeated 1000 times over, with noise added.
train_noise_stddev = 0.1
N = 9
d = 1
original_data = np.array([[-0.5], [0.0], [0.5]])
clean_data = np.tile(original_data, (N/3, 1))
np.random.shuffle(clean_data)
noisy_data = clean_data + np.random.normal(size = clean_data.shape,
                                           scale = train_noise_stddev)


## -----------------------------------
## Fit the model to the training data.
## -----------------------------------

batch_size = min(100,noisy_data.shape[0])

method = 'gradient_descent_stages'

if method == 'gradient_descent':
    n_epochs = 50000
    learning_rate = 1.0e-2
    import dae_train_gradient_descent
    dae_train_gradient_descent.fit(mydae,
                                   X = clean_data,
                                   noisy_X = noisy_data,
                                   batch_size = batch_size,
                                   n_epochs = n_epochs,
                                   learning_rate = learning_rate,
                                   verbose = True)
elif method == 'gradient_descent_stages':
    n_epochs = (30000,10000,10000,10000)
    learning_rate = (.04,.02,.01,.005)
    import dae_train_gradient_descent
    for (n_ep,lr) in zip(n_epochs,learning_rate):
         print "learning_rate = %f for %d epochs"%(lr,n_ep)
         dae_train_gradient_descent.fit(mydae,
                                       X = clean_data,
                                       noisy_X = noisy_data,
                                       batch_size = batch_size,
                                       n_epochs = n_ep,
                                       learning_rate = lr,
                                       verbose = True)
else:
    print "unknown method: %s"%method


mydae.set_params_to_best_noisy()
# mydae.set_params_to_best_noiseless()

n_error = 0
n = 0
for (x,corr_x) in zip(clean_data[0:20],noisy_data[0:20]):
    n+=1
    xx=x[0]
    xc=corr_x[0]
    r = mydae.encode_decode(corr_x.reshape(1,d))[0,0]
    error = np.sign(xc-xx)!=np.sign(xc-r)
    print xx,xc,r,error
    n_error+=error
print "GRADIENT SIGN ERROR RATE = ",n_error/float(n)
   

quit()

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
A = mydae.logging['noiseless']['mean_abs_loss'] < 4.0 * mydae.logging['noiseless']['mean_abs_loss'][-1]
if np.any(A):
    interesting_index = np.where( A )[0][0]
else:
    interesting_index = int(n_epochs / 10)
del A
print "interesting_index = %d" % interesting_index

plot_training_loss_history(mydae, os.path.join(output_directory, 'absolute_losses_start_to_interesting_index.png'), interesting_index)
plot_training_loss_history(mydae, os.path.join(output_directory, 'absolute_losses_start_to_end.png'), -1)


##########################################
##                                      ##
## plotting the reconstruction function ##
##                                      ##
##########################################

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

if method == 'hmc':
    hyperparams_contents_for_method = """
    <p>frogleap jumps  : %d</p>
    <p>epsilon : %f</p>
    """ % (L, epsilon)
elif method == 'gradient_descent' or method == 'gradient_descent_multi_stage':
    hyperparams_contents_for_method = """
    <p>learning rate  : %f</p>
    """ % (learning_rate,)

hyperparams_contents = """
<p>nbr visible units : %d</p>
<p>nbr hidden  units : %d</p>

<p>batch size : %d</p>
<p>epochs : %d</p>

%s

<p>training noise : %0.6f</p>

<p>dataset points : %d</p>
<p>dataset noise : %0.6f</p>
<p>angle restriction : %0.6f</p>
""" % (mydae.n_inputs,
       mydae.n_hiddens,
       batch_size,
       n_epochs,
       hyperparams_contents_for_method,
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


