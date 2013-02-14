
import numpy as np
# import scipy

def sample_trajectory_1D(x, y, kernel, x_star, obs_noise_stddev, n_samples = None):
    """
    x is of shape (N,) and contains the domain points
    
    y is of shape (N,) and contains the values at the domain points x
    
    kernel is a function that takes a two arguments x1 and x2
    with shapes (N1,) and (N2,) respectively and returns an array
    of shape (N1, N2) with usually something like 0.5*(x1[n1] - x2[n2])**2

    This function returns trajectory points for the values
    at x_star. At this moment I'm not sure if points from x
    are barred from being included in x_star.
    """

    assert len(x.shape) == 1
    N = x.shape[0]
    assert N == y.shape[0]

    assert len(x_star.shape) == 1
    M = x_star.shape[0]

    kxx = kernel(x,x)
    kxxs = kernel(x,x_star)
    kxsx = kernel(x_star,x)
    kxsxs = kernel(x_star,x_star)

    assert (N,N) == kxx.shape
    assert (N,M) == kxxs.shape
    assert (M,N) == kxsx.shape
    assert (M,M) == kxsxs.shape

    A = np.linalg.inv(kxx + obs_noise_stddev**2)
    B = kxsx.dot(A)

    f_star_mean = B.dot(y)
    f_star_cov = kxsxs - B.dot(kxxs)
    
    if n_samples:
        samples = np.random.multivariate_normal(mean = f_star_mean, cov = f_star_cov, size = n_samples)
    else:
        samples = np.random.multivariate_normal(mean = f_star_mean, cov = f_star_cov)

    return {'samples' : samples,
            'f_star_mean': f_star_mean,
            'f_star_cov': f_star_cov}

