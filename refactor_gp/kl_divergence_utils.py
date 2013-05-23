
import numpy as np



#### WORK IN PROGRESS ####




##########################

### Simple one dimensional case ###

import scipy
import scipy.optimize

def compute_kernel_variance(x):
    assert len(x.shape) == 1
    d = x.shape[0]
    
    # split as train/valid
    # perform cross-validation to just average the results
    # (or pick the maximal value so we don't get bad surprizes)


def compute_kernel_variance_helper(x_train, x_valid):

    def U(x):
        
             best_q = scipy.optimize.fmin_cg(U, q0,
                                            gtol=1e-8,
                                            maxiter = 50)
 