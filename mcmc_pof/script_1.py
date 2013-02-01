
import numpy as np

import ninja_star_distribution



proposal_stddev = 0.1
n_samples = 500
thinning_factor = 1000
burn_in = n_samples * thinning_factor / 10

mcmc_method = 'metropolis_hastings_E'
#mcmc_method = 'metropolis_hastings_grad_E'

import metropolis_hastings_sampler
# Don't start right on the origin because strange
# things can happen with the polar coordinates there.
x0 = np.random.normal(size=(2,))
symmetric_proposal = lambda x: x + np.random.normal(size=x.shape, scale = proposal_stddev)

if mcmc_method == 'metropolis_hastings_E':
    (X, acceptance_ratio) = metropolis_hastings_sampler.run_chain_with_energy(ninja_star_distribution.E, x0, symmetric_proposal, n_samples, thinning_factor = thinning_factor, burn_in = burn_in)
elif mcmc_method == 'metropolis_hastings_grad_E':
    (X, acceptance_ratio) = metropolis_hastings_sampler.run_chain_with_energy(None, x0, symmetric_proposal, n_samples, thinning_factor = thinning_factor, burn_in = burn_in, grad_E = ninja_star_distribution.grad_E)
else:
    error("Unrecognized value for parameter 'mcmc_method' : %s" % (mcmc_method,))



print "Got the samples. Acceptance ratio was %f" % acceptance_ratio


# Implement metropolis_hastings_sampler.run_chain_with_energy
# with grad_E instead.

if True:
    import matplotlib
    matplotlib.use('Agg')
    import pylab
    import matplotlib.pyplot as plt

    pylab.hold(True)
    pylab.scatter(X[:,0], X[:,1])

    print "Computing the original pdf values."
    M = 4.0
    mesh_x,mesh_y = np.mgrid[-M:M:.01, -M:M:.01]
    z = ninja_star_distribution.mesh_pdf(mesh_x, mesh_y)

    print "Generating the nice plots."
    model_pdf_values_plot_handle = plt.pcolor(mesh_x, mesh_y, z)
    #plt.winter()
    plt.pink()
    #d = plt.colorbar(model_pdf_value_plot_handle, orientation='horizontal')

    pylab.draw()
    id_number = int(np.random.uniform(0,10000))
    outfile = "/u/alaingui/umontreal/tmp/script_1_%s_%0.5d.png" % (mcmc_method, id_number)
    pylab.savefig(outfile, dpi=100)
    pylab.close()
    print "Wrote %s" % (outfile,)

    quit()
