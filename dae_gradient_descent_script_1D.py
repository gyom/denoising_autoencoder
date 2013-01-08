#!/usr/bin/env python

import numpy as np
np.random.seed(38734)

import dae
import debian_spiral
import numpy as np

np.random.seed(99)

#dae = reload(dae)
mydae = dae.DAE(n_inputs=1,
                n_hiddens=20,
                output_scaling_factor=5.0)

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
    learning_rate = 1.0e-5
    import dae_train_gradient_descent
    dae_train_gradient_descent.fit(mydae,
                                   X = clean_data,
                                   noisy_X = noisy_data,
                                   batch_size = batch_size,
                                   n_epochs = n_epochs,
                                   learning_rate = learning_rate,
                                   verbose = True)
elif method == 'gradient_descent_stages':
    n_epochs = (1000,2000,3000,10000)
    learning_rate = (1e-3,1e-4,3e-5,1e-5)
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
   

#quit()

## --------------------------------------
## Produce a report of the trained model.
## --------------------------------------

import os

# create a new directory to host the result files of this experiment
if os.getenv("DENOISING_REPO")=="":
   print "Please define DENOISING_REPO environment variable"
   quit()
output_directory = os.getenv("DENOISING_REPO")+'/denoising_autoencoder/plots/experiment_%0.6d' % int(np.random.random() * 1.0e6)
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
elif method == 'gradient_descent' or method == 'gradient_descent_stages':
    hyperparams_contents_for_method = """
    <p>learning rate  : %s </p>
    """ % (str(learning_rate),)
else:
    error("Unknown value for method.")

hyperparams_contents = """
<p>nbr visible units : %d</p>
<p>nbr hidden  units : %d</p>

<p>batch size : %d</p>
<p>epochs : %s</p>
%s
<p>training noise : %f</p>
""" % (mydae.n_inputs,
       mydae.n_hiddens,
       batch_size,
       str(n_epochs),
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


