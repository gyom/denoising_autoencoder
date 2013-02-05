
import numpy as np

ring_radius = 2.0
log_offset = 1.0

def E(X):
    """
    X of shape (N,2) or (2,)

    Returns a result of shape (N,) or just a scalar.
    """

    if len(X.shape) == 1:
        x = X[0]
        y = X[1]
    else:
        x = X[:,0]
        y = X[:,1]

    return ((np.sqrt(x**2 + y**2) - ring_radius)**2   +
            (x**2 - y**2)**2 +
            0.5 * np.log(x**2 + y**2 + log_offset) )

def grad_E(X):
    """
    X of shape (N,2) or (2,)

    Returns a result of shape (N,2) or (2,).
    """

    if len(X.shape) == 1:
        x = X[0]
        y = X[1]
    else:
        x = X[:,0]
        y = X[:,1]

    # Almost the same thing in the two components, but with an x changed to y.

    return np.hstack([(2*(np.sqrt(x**2 + y**2) - ring_radius)/np.sqrt(x**2 + y**2)*x   +
                       4*(x**2 - y**2)*x +
                       x/(x**2 + y**2 + log_offset) ),
                      (2*(np.sqrt(x**2 + y**2) - ring_radius)/np.sqrt(x**2 + y**2)*y   +
                       4*(x**2 - y**2)*(-y) +
                       y/(x**2 + y**2 + log_offset) )])


def mesh_pdf(mesh_x, mesh_y):
    """
    M = 5.0
    mesh_x,mesh_y = np.mgrid[-M:M:.01, -M:M:.01]
    """
    x = mesh_x
    y = mesh_y
    
    z_energy = ((np.sqrt(x**2 + y**2) - ring_radius)**2   +
                (x**2 - y**2)**2 +
                0.5 * np.log(x**2 + y**2 + log_offset) )
    z = np.exp( - z_energy )

    # We will attempt here to normalize the density.
    intz = z.sum()
    delta_x = mesh_x[1][0] - mesh_x[0][0]
    delta_y = mesh_y[0][1] - mesh_y[0][0]

    return (z / intz * delta_x * delta_y)



def compute_normalizing_constant():
    M = 5.0
    mesh_x,mesh_y = np.mgrid[-M:M:.01, -M:M:.01]

    x = mesh_x
    y = mesh_y
    
    z_energy = ((np.sqrt(x**2 + y**2) - ring_radius)**2   +
                (x**2 - y**2)**2 +
                0.5 * np.log(x**2 + y**2 + log_offset) )
    z = np.exp( - z_energy )

    # We will attempt here to normalize the density.
    intz = z.sum()
    delta_x = mesh_x[1][0] - mesh_x[0][0]
    delta_y = mesh_y[0][1] - mesh_y[0][0]

    return intz * delta_x * delta_y



def cross_entropy(X):
    """
    See p.88 of Kevin Murphy's book.
    """

    if len(X.shape) == 1:
        x = X[0]
        y = X[1]
    else:
        x = X[:,0]
        y = X[:,1]

    normalizing_constant = compute_normalizing_constant()

    energy_values = ((np.sqrt(x**2 + y**2) - ring_radius)**2   +
                (x**2 - y**2)**2 +
                0.5 * np.log(x**2 + y**2 + log_offset) )

    return energy_values.mean() + np.log(normalizing_constant)




def main():
    import metropolis_hastings_sampler
    sampling_options = {}
    sampling_options["E"] = E
    sampling_options["proposal_stddev"] = 0.1
    sampling_options["thinning_factor"] = 10000
    sampling_options["n_samples"] = 10000
    sampling_options["mcmc_method"] = "metropolis_hastings_E"
    results = metropolis_hastings_sampler.mcmc_generate_samples(sampling_options)


    import cPickle
    output_pkl_name = "ninja_star_metropolis_hasting_samples_n_%d_thin_%d_prop_stddev_%f.pkl" % (sampling_options["n_samples"], sampling_options["thinning_factor"], sampling_options["proposal_stddev"])
    f = open(output_pkl_name, "w")
    cPickle.dump(results['samples'], f)
    f.close()

    print "Acceptance ratio : %f" % results["acceptance_ratio"]
    print "Wrote %s" % (output_pkl_name, )


if __name__ == "__main__":
    main()