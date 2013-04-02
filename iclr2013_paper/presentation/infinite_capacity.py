import numpy as np

def fit_cae_1D(x, p, noise_stddev):
    n = x.shape[0]
    assert n == p.shape[0]

    delta = x[1:] - x[:-1]
    assert(np.all(delta > 0))

    eqn_R = np.zeros((n,n))
    b = np.zeros((n,))

    # eqn_R . r + b = 0

    for i in range(0,n):

        if i < n-1:
            eqn_R[i, i] += p[i]*delta[i]
            b[i] += -p[i]*delta[i]*x[i]
            eqn_R[i, i]   +=  p[i]/delta[i]*noise_stddev**2
            eqn_R[i, i+1] += -p[i]/delta[i]*noise_stddev**2
   
        if 0 < i:
            eqn_R[i, i]   +=  p[i-1]/delta[i-1]*noise_stddev**2
            eqn_R[i, i-1] += -p[i-1]/delta[i-1]*noise_stddev**2

    # print eqn_R
    # print b

    r = np.linalg.solve(eqn_R, -b)
    return r



def fit_dae_1D(x, p, noise_stddev):

    def fit_one_point(y):
        w = np.exp(-0.5*(x - y)**2/noise_stddev**2)
        #print w
        return (w * p * x).sum() / (w * p).sum()

    return np.array([fit_one_point(y) for y in x])



def fit_dae(X, P, noise_stddev):
    """
    X has shape (N,d)
    P has shape (N,)
    """

    print X.shape
    (N,d) = X.shape
    assert len(P.shape) == 1 and P.shape[0] == N

    def fit_one_point(y):
        w = np.exp(-0.5*((X - y)**2).sum(axis=1)/noise_stddev**2)
        assert(len(w.shape) == 1 and w.shape[0] == N)

        S = np.tile( (w * P).reshape((N,1)), (1,d))

        return (S * X).sum(axis=0) / S[:,0].sum()

    return np.vstack([fit_one_point(y) for y in X])


#
## TODO : This function is in a bad state of disrepair.
#
def fit_cae_2D(meshX, meshY, meshP, noise_stddev):
    """
    meshX has shape (d1,d2,...,dN)
    meshY has shape (d1,d2,...,dN)
    meshP has shape (d1,d2,...,dN)
    """

    D = len(meshX.shape)
    #print meshX.shape
    assert meshP.shape == meshX.shape
    assert meshP.shape == meshY.shape

    def collapse(R1, R2):
        return np.hstack((R1.reshape((-1,)), R2.reshape((-1,))))

    def expand(A_collapsed):
        N = A_collapsed.shape[0]
        R1_collapsed = A_collapsed[0:(N/2)]
        R2_collapsed = A_collapsed[(N/2):]
        return (R1_collapsed.reshape(meshX.shape), R2_collapsed.reshape(meshX.shape))

    # alright, screw that, I'm only dealing with D == 2 here
    assert D == 2
    delta = meshX[0,1] - meshX[0,0]
    assert delta == meshX[1,0] - meshX[0,0]

    import scipy

    def U(r):
        (R1,R2) = expand(r)
        a = (((meshr - meshX)*meshP*delta)**2).sum()
        b0 = ( ((meshr[1:,:] - meshr[:-1,:])**2)*meshP[:-1,:]/delta  ).sum()*noise_stddev**2
        b1 = ( ((meshr[:, 1:] - meshr[:, :-1])**2)*meshP[:, :-1]/delta  ).sum()*noise_stddev**2
        return a + b0 + b1

    meshr = expand( scipy.optimize.fmin_cg(U, collapse(np.zeros(meshX.shape))) )
    return meshr


    # i = 0
    # eqn_R[i, i] += p[i]*delta[i]
    # b[i] += -p[i]*delta[i]*x[i]
    # eqn_R[i, i]   +=  p[i]/delta[i]
    # eqn_R[i, i+1] += -p[i]/delta[i]
    # 
    # for i in range(1,n-1):
    #
    #    eqn_R[i, i] += p[i]*delta[i]
    #     b[i] += -p[i]*delta[i]*x[i]
    #
    #     eqn_R[i, i]   +=  p[i]/delta[i]
    #     eqn_R[i, i+1] += -p[i]/delta[i]
    #
    #     eqn_R[i, i]   +=  p[i-1]/delta[i-1]
    #     eqn_R[i, i-1] += -p[i-1]/delta[i-1]
    #
    # i = n-1
    # eqn_R[i, i]   +=  p[i-1]/delta[i-1]
    # eqn_R[i, i-1] += -p[i-1]/delta[i-1]