
import numpy as np
import cPickle

A = cPickle.load(open("/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d10_eig0.1_comp25_001/train_samples_extra.pkl"))

component_covariances = A['component_covariances']
component_means = A['component_means']

print component_means

import refactor_gp
import refactor_gp.gyom_utils
from   refactor_gp.gyom_utils import mvnpdf

data = cPickle.load(open("/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d10_eig0.1_comp25_001/train_samples.pkl"))

print component_means.shape
print data.shape

(K, D) = component_means.shape
(N, _) = data.shape
assert ( (N,D) == data.shape )

# DEVEL
#N=10

B = np.zeros((N, K))

for n in range(N):
	for k in range(K):
		if n % 100 == 0:
			print n
		B[n, k] = mvnpdf(data[n,:], component_means[k], component_covariances[k])

# now C contains the value of p(x) for each x in data
# because we computed the mean over the K components
C = B.mean(axis=1)

#print "B"
#print B
#print "C"
#print C

print "np.log(C).mean()"
print np.log(C).mean()