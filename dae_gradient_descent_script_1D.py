#!/usr/bin/env python

import numpy as np
np.random.seed(38734)

import dae
#dae = reload(dae)
mydae = dae.DAE(n_inputs=1,
                n_hiddens=20,
                output_scaling_factor=5.0)

## ----------------------
## Get the training data.
## ----------------------

# Three training points, repeated 1000 times over, with noise added.
train_noise_stddev = 0.1
N = 3000
d = 1
original_data = np.array([[-1.0], [0.0], [2.0]])
clean_data = np.tile(original_data, (N/3, d))
np.random.shuffle(clean_data)
noisy_data = clean_data + np.random.normal(size = clean_data.shape,
                                           scale = train_noise_stddev)


## -----------------------------------
## Fit the model to the training data.
## -----------------------------------

batch_size = 100
n_epochs = 10000

method = 'gradient_descent'

if method == 'gradient_descent':
    import dae_train_gradient_descent
    learning_rate = 1.0e-5
    dae_train_gradient_descent.fit(mydae,
                                   X = clean_data,
                                   noisy_X = noisy_data,
                                   batch_size = batch_size,
                                   n_epochs = n_epochs,
                                   learning_rate = learning_rate,
                                   verbose = True)
else:
    error("Unrecognized training method.")


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
A = mydae.logging['noiseless']['mean_abs_loss'] < 4.0 * mydae.logging['noiseless']['mean_abs_loss'][-1]
if np.any(A):
    interesting_index = np.where( A )[0][0]
else:
    interesting_index = int(n_epochs / 10)
if interesting_index == 0:
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

def plot_reconstruction_function(mydae, outputfile, n_buckets = 30, domain_start = -3.0, domain_end = 3.0):

    domain_values = np.linspace(domain_start, domain_end, n_buckets)

    print "Making predictions for the grid."
    r_values = mydae.encode_decode(domain_values.reshape((-1,1)))

    print "Generating plot."
    pylab.hold(True)

    p1 = pylab.plot(domain_values, r_values, c='#f9a21d')
    p2 = pylab.plot(domain_values, domain_values, c='#1861cd')
    
    pylab.axis([domain_start, domain_end, domain_start, domain_end])
    pylab.legend([p1,p2], ["$r(x)$", "$x$"])
    pylab.draw()
    pylab.savefig(outputfile, dpi=300)
    pylab.close()




plot_reconstruction_function(mydae, os.path.join(output_directory, 'rx_versus_x.png'),
                              n_buckets = 1000,
                              domain_start = -3.0, domain_end = 3.0)


plot_reconstruction_function(mydae, os.path.join(output_directory, 'rx_versus_x_zoomed_on_origin.png'),
                              n_buckets = 1000,
                              domain_start = -0.1, domain_end = 0.1)



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
else:
    error("Unknown value for method.")

hyperparams_contents = """
<p>nbr visible units : %d</p>
<p>nbr hidden  units : %d</p>

<p>batch size : %d</p>
<p>epochs : %d</p>
%s
<p>training noise : %f</p>
""" % (mydae.n_inputs,
       mydae.n_hiddens,
       batch_size,
       n_epochs,
       hyperparams_contents_for_method,
       train_noise_stddev)

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
    <img src='rx_versus_x_zoomed_on_origin.png' width='600px'/>
    <img src='rx_versus_x.png' width='600px'/>
</div>

</body>
</html>""" % (hyperparams_contents,
              params_contents)

f.write(contents)
f.close()


