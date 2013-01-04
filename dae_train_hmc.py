
import numpy as np
import hamiltonian_monte_carlo as hmc

def perform_one_update(the_dae, X, noise_stddev, L, epsilon, simulate_only  = False):

    # We'll format the parameters of the DAE so that they
    # fit into the mold of hamiltonian_monte_carlo.one_step_update,
    # we'll call that function and then reformat the parameters.
    #
    # In particular, this means that we'll transform (W, b, c)
    # into the vector of position for the Hamiltonian Monte Carlo.

    # Helper functions. One being the opposite of the other.
    def serialize_params_as_q(W, b, c):
        return np.hstack((W.reshape((-1,)),
                          b.reshape((-1,)),
                          c.reshape((-1,)))).reshape((-1,))

    def read_params_from_q(q, n_inputs, n_hiddens):
        n_elems_W = n_inputs * n_hiddens
        n_elems_b = n_inputs
        n_elems_c = n_hiddens

        W = q[0 : n_elems_W].reshape((n_inputs, n_hiddens)).copy()
        b = q[n_elems_W : n_elems_W + n_elems_b].reshape((n_elems_b,)).copy()
        c = q[n_elems_W + n_elems_b : n_elems_W + n_elems_b + n_elems_c].reshape((n_elems_c,)).copy()
        
        return (W, b, c)

    # We'll reuse the same perturbed_X for all the evaluations
    # of the potential during the leapfrog.
    if noise_stddev > 0.0:
        perturbed_X = X + np.random.normal(scale=noise_stddev, size=X.shape)
    else:
        perturbed_X = X.copy()

    def U(q):
        # the dataset X is baked into the definition of U(q)
        (W, b, c) = read_params_from_q(q, the_dae.n_inputs, the_dae.n_hiddens)
        (loss, _, _) = the_dae.theano_loss(W, b, c, perturbed_X, X)
        return loss

    def grad_U(q):
        (W, b, c) = read_params_from_q(q, the_dae.n_inputs, the_dae.n_hiddens)
        (grad_W, grad_b, grad_c) = the_dae.theano_gradients(W, b, c, perturbed_X, X)
        # We can use the same function to spin the gradients into the
        # momentum vector. This works well because
        #   grad_W.shape == W.shape
        #   grad_b.shape == b.shape
        #   grad_c.shape == c.shape
        return serialize_params_as_q(grad_W, grad_b, grad_c)


    current_q = serialize_params_as_q(the_dae.W, the_dae.b, the_dae.c)
    updated_q = hmc.one_step_update(U, grad_U, epsilon, L, current_q, verbose=False)

    # Our function U(q) is mutating the dae in a way that makes
    # it necessary that we rewrite the parameters values at the
    # end even if current_q == updated_q.
    (the_dae.W, the_dae.b, the_dae.c) = read_params_from_q(updated_q, the_dae.n_inputs, the_dae.n_hiddens)


def fit(the_dae, X, batch_size, n_epochs, noise_stddev, L, epsilon, verbose=False):
    """
    Fit the model to the data X.
    
    Parameters
    ----------
        X: array-like, shape (n_examples, n_inputs)
        Training data, where n_examples in the number of examples
        and n_inputs is the number of features.

        L: int, number of leapfrog steps to do
        epsilon: double, size of leapfrog steps
    """

    # We don't necessarily want to reset the parameters before fitting.
    # the_dae.reset_params()

    # We'll be using double indirection to shuffle
    # around the minibatches. We will keep shuffling
    # the indices in 'inds' and the chunks will be
    # described by 'inds_ranges' which will be a collection
    # of ranges.
    #
    # ex :
    #    self.batch_size is 3
    #    inds is [10,3,2,5,9,8,6,0,7,4,1]
    #    inds_ranges is [(0,3), (3,6), (6,9), (9,10)]
    #
    # Results in batches being
    #    [10,3,2], [5,9,8], [0,7,4], [1]
    
    inds = range(X.shape[0])
        
    n_batches = len(inds) / batch_size
    inds_ranges = []
    for k in range(0, n_batches):

        start = k * batch_size
        if start >= X.shape[0]:
            break

        end = (k+1) * batch_size
        end = min(end, X.shape[0])

        # Keep in mind that the lower bound is inclusive
        # and the upper bound is exclusive.
        # This is why 'end' terminates with X.shape[0]
        # while that value would be illegal for 'start'.

        inds_ranges.append( (start, end) )
            
    if verbose:
        print "The ranges used for the minibatches are "
        print inds_ranges

    for epoch in range(n_epochs):

        # Shuffle the 'inds', because we don't modify
        # the 'inds_ranges'. Only one of them has to change.
        np.random.shuffle(inds)
            
        for (start, end) in inds_ranges:
            X_minibatch = X[inds[start:end]]
            # calling the function defined here in
            # dae_train_gradient_descent.perform_one_update
            perform_one_update(the_dae, X_minibatch, noise_stddev, L, epsilon)

        if the_dae.want_logging:
            if verbose and (epoch % 100 == 0):
                sys.stdout.flush()
                print "Epoch %d" % epoch
                the_dae.perform_logging(X, noise_stddev, verbose = True)
            else:
                the_dae.perform_logging(X, noise_stddev, verbose = False)
