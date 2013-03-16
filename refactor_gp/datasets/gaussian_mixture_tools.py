#!/bin/env python

import numpy as np
import scitools

import sys, os, time

import refactor_gp
import refactor_gp.gyom_utils
from   refactor_gp.gyom_utils import conj


def generate_a_covariance_matrix(target_v, ratio_of_other_eigenvalues):
    """
    target_v is the desired leading eigenvector multiplied by its eigenvalue.
    We usually get it by subtracting one point from the next on
    a manifold that we are "tesselating".

    We can infer from 'target_v' the leading eigenvalue.
    As for the other ones, we'll select them to be such that
    they are 'ratio_of_other_eigenvalues' times that leading eigenvalue.
    """

    assert len(target_v.shape) == 1
    d = len(target_v)

    # looping just in case we randomnly draw a vector
    # that has the same orientation as the target_v
    done = False
    while not done:
        E = np.random.normal(size=(d,d))
        E[0,:] = target_v
        try:
            B = scitools.numpyutils.Gram_Schmidt(E, normalize=True)
            done = True
        except:
            pass

    # make sure that the leading vector is correct
    # and that the Gram_Schmidt operation didn't reorder
    # the vectors
    #assert np.all(  B[0,:] / np.linalg.norm(B[0,:]) == target_v / np.linalg.norm(target_v) )
    if not np.all(
        np.abs(
            B[0,:] / np.linalg.norm(B[0,:]) - target_v / np.linalg.norm(target_v)
            ) < 1.0e-8):
        print B[0,:] / np.linalg.norm(B[0,:])
        print target_v / np.linalg.norm(target_v)
        print "Misunderstanding with the Gram-Schmidt normalization."
        print "The leading vector should be kept intact, but it wasn't."
        quit()

    V = B
    leading_eigval = np.linalg.norm(target_v)
    eigvals = np.ones(d,) * leading_eigval * ratio_of_other_eigenvalues
    eigvals[0] = leading_eigval

    # debug
    # D = np.diag(np.sqrt(1/eigvals))
    # D = np.diag(1/eigvals**2)
    # D = np.diag(eigvals)
    #
    # This one is the correct way.
    D = np.diag(eigvals**2)
    

    covariance_matrix = np.linalg.inv(V).dot(D).dot(V)

    #print "\n V is",
    #print V
    #print "\n D is",
    #print D

    #print "\ndet of cov matrix is %f" % np.linalg.det(covariance_matrix)
    assert np.linalg.det(covariance_matrix) != 0.0

    return covariance_matrix


def generate_collection_covariance_matrices(E, ratio_eigvals):

    n, d = E.shape
    covariance_matrices = np.zeros((n, d, d))

    for i in range(n):
        # Compute v for all iterations except the last.
        # Reuse the previous v for the last iteration.
        if i < n-1:
            v = E[i+1,:] - E[i,:]
        #print "Want to generate a covariance matrix with leading eigvec in the direction of"
        #print v
        covariance_matrices[i,:,:] = generate_a_covariance_matrix(v, ratio_eigvals)

    return covariance_matrices


#def sample_from_mixture(component_means, component_covariances, n_samples):
#
#    assert component_means != None
#    assert component_covariances != None
#
#    (n_components, d) = component_means.shape
#    (n_components1, d1, d2) = component_covariances.shape
#    assert n_components == n_components1
#    assert d == d1
#    assert d == d2
#
#    samples = np.zeros((n_samples, d))
#    component_indices = np.zeros((n_samples,))
#    for k in np.arange(n_samples):
#        c = np.random.randint(n_components)
#        component_indices[k] = c
#        samples[k,:] = np.random.multivariate_normal(mean=component_means[c,:], cov=component_covariances[c,:,:])
#        
#    return (samples, component_indices)

import refactor_gp
import refactor_gp.gyom_utils
from refactor_gp.gyom_utils import mvnpdf
from refactor_gp.gyom_utils import grad_mvnpdf
from refactor_gp.gyom_utils import normalized_weighted_sum_with_log_coefficients

#def pdf(x, component_means, component_covariances):
#
#    d = x.shape[0]
#    assert len(x.shape) == 1
#    assert component_means != None
#    assert component_covariances != None
#
#    (n_components, d1) = component_means.shape
#    (n_components1, d2, d3) = component_covariances.shape
#    assert n_components == n_components1
#    assert (d,d,d) == (d1,d2,d3)
#
#    return np.array([mvnpdf(x,component_means[k,:],component_covariances[k,:,:]) for k in np.arange(n_components)]).mean()


#def grad_pdf(x, component_means, component_covariances = None):
#
#    d = x.shape[0]
#    assert len(x.shape) == 1
#    assert component_means != None
#    assert component_covariances != None
#
#    (n_components, d1) = component_means.shape
#    (n_components1, d2, d3) = component_covariances.shape
#    assert n_components == n_components1
#    assert (d,d,d) == (d1,d2,d3)
#
#    # Stack all the contributions from all the components in n_components rows.
#    # Each row has d coefficients.
#    # Collapse the columns by averaging.
#    A = np.vstack([grad_mvnpdf(x,component_means[k,:],component_covariances[k,:,:]) for k in np.arange(n_components)])
#    assert (n_components, d)== A.shape
#    return A.mean(axis=0)

#def grad_E(x, component_means, component_covariances):
#    return - grad_pdf(x, component_means, component_covariances) / pdf(x, component_means, component_covariances)


class MixtureMVN():
    def __init__(self, means, covariances):
        (K,d) = means.shape
        (K0,d1,d2) = covariances.shape
        assert (K==K0) and (d==d1) and (d==d2)
        self.d = d
        self.K = K
        self.means = means
        self.covariances = covariances
        self.precisions = np.zeros(covariances.shape)
        self.precisions_dets = np.zeros((K,))
        for k in range(0,K):
            self.precisions[k,:,:] = np.linalg.inv(covariances[k,:,:])
            self.precisions_dets[k] = np.linalg.det(self.precisions[k,:,:])

    def grad_pdf(self, x):
        assert self.d == x.shape[0]
        assert len(x.shape) == 1

        logc = np.zeros((self.K,))
        E = np.zeros((self.K,self.d))
        for k in range(self.K):
            (logck, Ek) = grad_mvnpdf(x, self.means[k,:], precision=self.precisions[k,:,:], precision_det=self.precisions_dets[k], want_log_decomposition=True)
            logc[k] = logck
            E[k,:] = Ek

        res = normalized_weighted_sum_with_log_coefficients(logc, E, axis=0)
        assert res.shape == x.shape
        return res

    def sample(self, n_samples, want_indices=False):

        samples = np.zeros((n_samples, self.d))
        component_indices = np.zeros((n_samples,))
        for n in np.arange(n_samples):
            c = np.random.randint(self.K)
            component_indices[n] = c
            samples[n,:] = np.random.multivariate_normal(mean=component_means[c,:], cov=component_covariances[c,:,:])

        if want_indices:
            return (samples, component_indices)
        else:
            return samples




# If this is called, we'll just run a sanity check.
def main():

    target_v = np.array([1,1])
    ratio_eig = 0.1
    cov = generate_a_covariance_matrix(target_v, ratio_eig)

    print "\ncovariance matrix generated"
    print cov

    U, s, V = np.linalg.svd(cov)
    print "\nsvd decomposition"
    print s
    print V

    #w,vh = np.linalg.eig(cov)
    #print "decomposition "
    #print w, vh

    n_samples = 1000
    samples = np.random.multivariate_normal(mean = np.array([0, 0]), cov = cov, size=n_samples)

    #print samples.T.dot(samples)
    cov_estimate = 1.0/n_samples * samples.T.dot(samples)
    print "\nexperimental covariance"
    print cov_estimate


if __name__ == "__main__":
    main()



