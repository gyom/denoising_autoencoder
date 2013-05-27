#!/bin/env python

import numpy as np
import os, sys, time

import refactor_gp
import refactor_gp.models
from   refactor_gp.models import dae_untied_weights

import refactor_gp.datasets
from   refactor_gp.datasets import gaussian_mixture_tools

#import refactor_gp
import refactor_gp.sampling
from   refactor_gp.sampling import dispatcher

import refactor_gp.tools
from   refactor_gp.tools import plot_overview_slices_for_samples

def usage():
    print ""

def main(argv):
    """

    """

    import getopt
    import cPickle
    import json

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["pickled_dae_dir=", "exact_grad_E_from_mixture_mvn_pickle=", "fake_train_stddev=", "mcmc_method=", "n_samples=", "thinning_factor=", "burn_in=", "proposal_stddev=", "langevin_stddev=", "langevin_beta=", "temperature=", "output_dir=", "n_E_approx_path=", "proposal_noise_scheme=", "want_overview_plots=", "omit_asymmetric_proposal_factor="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    pickled_dae_dir = None
    exact_grad_E_from_mixture_mvn_pickle = None
    fake_train_stddev = None
    mcmc_method = None
    n_samples = None
    thinning_factor = None
    burn_in = None
    proposal_stddev = None
    output_dir = None
    n_E_approx_path = None
    proposal_noise_scheme = None
    langevin_stddev = None
    langevin_beta = None
    temperature = None
    omit_asymmetric_proposal_factor = None
    want_overview_plots = False

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--pickled_dae_dir"):
            pickled_dae_dir = a
        elif o in ("--exact_grad_E_from_mixture_mvn_pickle"):
            exact_grad_E_from_mixture_mvn_pickle = a
        elif o in ("--fake_train_stddev"):
            fake_train_stddev = float(a)
        elif o in ("--mcmc_method"):
            mcmc_method = a
        elif o in ("--n_samples"):
            n_samples = int(a)
        elif o in ("--thinning_factor"):
            thinning_factor = int(a)
        elif o in ("--burn_in"):
            burn_in = int(a)
        elif o in ("--langevin_stddev"):
            langevin_stddev = float(a)
        elif o in ("--langevin_beta"):
            langevin_beta = float(a)
        elif o in ("--temperature"):
            temperature = float(a)
        elif o in ("--proposal_stddev"):
            proposal_stddev = float(a)
        elif o in ("--n_E_approx_path"):
            n_E_approx_path = float(a)
        elif o in ("--proposal_noise_scheme"):
            proposal_noise_scheme = a
        elif o in ("--omit_asymmetric_proposal_factor"):
            omit_asymmetric_proposal_factor = ((a == "True") or (a == "true") or (a == "1"))
        elif o in ("--want_overview_plots"):
            want_overview_plots = ((a == "True") or (a == "true") or (a == "1"))
        elif o in ("--output_dir"):
            output_dir = a
        else:
            assert False, "unhandled option"

    assert not (mcmc_method == None)
    if mcmc_method in ['MH_langevin_grad_E', 'langevin_grad_E']:
        assert not ((langevin_stddev == None) and (langevin_beta == None))
 
    # There are two alternatives to load a DAE.
    # Either through pickled_dae_dir or exact_grad_E_fake_dae_dir.
    # ex : 
    #     pickled_dae_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d2/experiment_11/000000"
    #     exact_grad_E_from_mixture_mvn_pickle = "/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d2_eig0.1_comp25_001/train_samples_extra.pkl"

    if pickled_dae_dir is not None:
        assert os.path.exists(pickled_dae_dir)
        assert os.path.isdir(pickled_dae_dir)

        pickled_dae_file = os.path.join(pickled_dae_dir, "trained_dae.pkl")
        pickled_dae_extra_file = os.path.join(pickled_dae_dir, "extra_details.pkl")
        assert os.path.exists(pickled_dae_file)
        assert os.path.exists(pickled_dae_extra_file)

        # irrelevant values because we load a pickle anyways
        mydae = dae_untied_weights.DAE_untied_weights(n_inputs = 1,
                                                      n_hiddens = 1,
                                                      act_func = ['tanh', 'tanh'])
        mydae.load_pickle(pickled_dae_file)
        extra_details = cPickle.load(open(pickled_dae_extra_file, "r"))

        n_inputs = extra_details['n_inputs']
        n_hiddens = extra_details['n_hiddens']

        # At this point we have a compatibility problem.
        # The older method of proceeding was by reading the train_stddev
        # using extra_details['model_losses'] and extra_details['noise_stddevs']
        # but now these structures are more complicated.
        #
        # This next if block is about building a list L of pairs
        # that will read the information from both methods (old and new)
        # so we can feed it to the actual code right after.

        if ((type(extra_details['model_losses']) == list) and
            (type(extra_details['noise_stddevs']) == list)):
            # old
            L = zip(extra_details['model_losses'], extra_details['noise_stddevs'])

            # Here's an example of what extra_details contains.
            """
            {'act_func': [u'tanh', u'id'],
            'computational_cost_in_seconds': 373,
            'early_termination_occurred': True,
            'lbfgs_rank': 4,
            'maxiter': 1000,
            'model_losses': [0.88493400434004688,  0.085861144685915214,  0.014545763882193379,  0.0033357369919251481,  0.0003573931475000403,  3.3169528288705622e-05,  nan,  nan,  nan,  nan],
            'n_hiddens': 30,
            'n_inputs': 10,
            'noise_stddevs': [1.0,  0.2782559402207124,  0.07742636826811268,  0.02154434690031883,  0.005994842503189405,  0.0016681005372000581,  0.00046415888336127757,  0.00012915496650148825,  3.5938136638046215e-05,  9.99999999999998e-06],
            'train_samples_pickle': '/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d10_eig0.1_comp25_001/train_samples.pkl'}
            """

        else:
            # new

            L = zip(extra_details['model_losses']['train'],
                    [(e['target'] if e.has_key('target') else e['sampled']) for e in extra_details['noise_stddevs']['train']])

            """
            "act_func": [
            "sigmoid",
            "sigmoid"
            ],
            "computational_cost_in_seconds": 12221,
            "lbfgs_rank": 8,
            "maxiter": 100,
            "model_losses": {
                "gentle_valid": [
                    0.0028618150711059568,
                    0.0028551105499267579,
                    0.0028442161560058594, ...],
            "train": [
                    0.00053718544006347657,
                    0.00053839420318603514,
                    0.00054026546478271487, ...],
            "valid": [
                    0.0028635965347290041,
                    0.0028334093093872071,
                    0.0028069316864013671, ...]
            "wider_gentle_valid": [
                    0.0036463272094726562,
                    0.0047698848724365235, ...]
            },
            "noise_stddevs": {
                "gentle_valid": [
                    {
                        "sampled": 10.000000000000002,
                        "target": 10.000000000000002
                    },
                    {
                        "sampled": 8.85866790410083,
                        "target": 8.85866790410083
                    },
            """


        # Get the last value for which we don't have 'nan' as value.
        train_stddev = None
        for (loss, noise_stddev) in L:
            if np.isnan(loss):
                break
            else:
                train_stddev = noise_stddev


        assert train_stddev
        print "Obtaining grad_E from r and the fact that train_stddev %f" % (train_stddev,)

        dae_params = read_parameters_from_dae(mydae)
        r = dae_params['r']

        # don't forget the minus sign here !
        # remember that r(x) - x is propto  -1 * dE/dx
        def grad_E(x):
            return -1*(r(x)-x) / train_stddev**2

        x0 = np.zeros((n_inputs,))
        sampling_options = {'x0':x0,
                            'f_prime':dae_params['f_prime'],
                            'r':dae_params['r'],
                            'r_prime':dae_params['r_prime'],
                            'grad_E':grad_E,
                            'train_stddev':train_stddev}

    elif exact_grad_E_from_mixture_mvn_pickle is not None:
        assert os.path.exists(exact_grad_E_from_mixture_mvn_pickle)
        assert fake_train_stddev is not None
        assert fake_train_stddev > 0

        extra_props = cPickle.load(open(exact_grad_E_from_mixture_mvn_pickle, "r"))
        # This nets us a structure that looks like the following.
        #   extra_props = {'component_means':component_means,
        #                  'component_covariances':component_covariances,
        #                  'd':d,
        #                  'ratio_eigvals':ratio_eigvals,
        #                  'n_components':n_components,
        #                  'f_parameters':f_parameters,
        #                  'computational_cost_in_seconds':computational_cost_in_seconds}

        mixturemvn = gaussian_mixture_tools.MixtureMVN(extra_props['component_means'], extra_props['component_covariances'])
            
        def grad_E(x):
            res = - mixturemvn.grad_pdf(x)
            #base_truth_res = gaussian_mixture_tools.grad_E(x, extra_props['component_means'], extra_props['component_covariances'])
            #assert(np.all(np.abs(res - base_truth_res) < 1e-8))

            if np.any(np.isnan(res)):
                print "grad_E got argument"
                print x
                print "and returned"
                print res
                print "ERROR. grad_E just returned nan !"
                quit()
            return res

        def r(x):
            return x + fake_train_stddev**2 * grad_E(x)

        def r_prime(x):
            return np.eye(x.shape[0])

        # We always sample from the origin and this is fine
        # because we generated the data so that would have 
        # some density at the origins.
        x0 = np.zeros((extra_props['d'],))

        sampling_options = {'x0':x0,
                            'r':r,
                            'r_prime':r_prime,
                            'grad_E':grad_E,
                            'train_stddev':fake_train_stddev}

    else:
        raise("You have to use either the --pickled_dae_dir or --exact_grad_E_from_mixture_mvn_pickle option.")


    sampling_options['mcmc_method'] = mcmc_method
    sampling_options['n_samples'] = n_samples

    if burn_in is not None:
        sampling_options['burn_in'] = burn_in
    if thinning_factor is not None:
        sampling_options['thinning_factor'] = thinning_factor
    if proposal_stddev is not None:
        sampling_options['proposal_stddev'] = proposal_stddev
    if n_E_approx_path is not None:
        sampling_options['n_E_approx_path'] = n_E_approx_path
    if proposal_noise_scheme is not None:
        sampling_options['proposal_noise_scheme'] = proposal_noise_scheme
    if omit_asymmetric_proposal_factor is not None:
        sampling_options['omit_asymmetric_proposal_factor'] = omit_asymmetric_proposal_factor
    if langevin_stddev is not None:
        sampling_options['langevin_stddev'] = langevin_stddev
    if langevin_beta is not None:
        sampling_options['langevin_beta'] = langevin_beta
    if temperature is not None:
        sampling_options['temperature'] = temperature


    print sampling_options

    #### Perform the sampling. ####
    res = dispatcher.mcmc_generate_samples(sampling_options)
    #
    # res has keys : samples
    #                elapsed_time
    #                proposals_per_second
    #                acceptance_ratio
    #                noise_levels

    print ""
    print "Sampling took %d seconds." % (res['elapsed_time'],)
    print "Proposals per second : 10^%0.2f." % (np.log(res['proposals_per_second'])/np.log(10),)
    print "Acceptance ratio : %0.3f." % (res['acceptance_ratio'],)
    print ""

    #### Write out the results. ####

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print "Creating directory %s" % (output_dir,)

    samples_file = os.path.join(output_dir, "mcmc_samples.pkl")
    cPickle.dump(res['samples'], open(samples_file, "w"))
    print "Wrote samples in %s" % (samples_file,)

    sampling_extra_details = res
    # Get rid of the memory-intensive 'samples'
    # field because it was already written in mcmc_samples.pkl.
    del sampling_extra_details['samples']
    for (k,v) in sampling_options.items():
        assert not sampling_extra_details.has_key(k)
        if k not in ['r', 'r_prime', 'f', 'f_prime', 'grad_E', 'x0']:
            sampling_extra_details[k] = v
        
    if pickled_dae_dir is not None:
        sampling_extra_details['pickled_dae_dir'] = pickled_dae_dir
    if exact_grad_E_from_mixture_mvn_pickle is not None:
        sampling_extra_details['exact_grad_E_from_mixture_mvn_pickle'] = exact_grad_E_from_mixture_mvn_pickle
    if fake_train_stddev is not None:
        sampling_extra_details['fake_train_stddev'] = fake_train_stddev


    sampling_extra_pickle_file = os.path.join(output_dir, "sampling_extra_details.pkl")
    sampling_extra_json_file = os.path.join(output_dir, "sampling_extra_details.json")
    cPickle.dump(sampling_extra_details, open(sampling_extra_pickle_file, "w"))
    json.dump(sampling_extra_details, open(sampling_extra_json_file, "w"), sort_keys=True, indent=4, separators=(',', ': '))
    print "Wrote %s" % (sampling_extra_pickle_file,)
    print "Wrote %s" % (sampling_extra_json_file,)


    #### Copy the original DAE (if applicable). ####

    # We'll copy the original DAE here because we
    # want to make sure that it stays around when
    # we want to inspect the samples and check the
    # characteristics of the DAE.

    if pickled_dae_dir is not None:
        import shutil
        output_dir_subdir = os.path.join(output_dir, "trained_dae")
        if os.path.exists(output_dir_subdir):
            shutil.rmtree(output_dir_subdir)
            shutil.copytree(pickled_dae_dir, output_dir_subdir)
            print "Transfered trained DAE from %s to %s" % (pickled_dae_dir, output_dir_subdir)

    #### Option to generate plots. ####

    # This option will probably be left on when
    # we do a lot of debugging to get to the right
    # hyperparameters, and then left off when
    # we just want to launch a lot of tasks afterwards.

    if want_overview_plots:
        plot_overview_slices_for_samples.main([None, "--pickled_samples_file=%s" % (samples_file,)] )

    print "Done."





def read_parameters_from_dae(some_dae):
    """
    This be a tool to get the functions r, r_prime, f_prime
    from a dae like DAE_untied_weights.
    """
    def r(x):
        # only asserted because that's what we expect,
        # not asserted because it would produce some conceptual
        # problem
        assert len(x.shape) == 1
        return some_dae.encode_decode(x.reshape((1,-1))).reshape((-1,))
    
    def r_prime(x):
        return some_dae.jacobian_encode_decode(x)
    
    def f(x):
        assert len(x.shape) == 1
        return some_dae.encode(x.reshape((1,-1))).reshape((-1,))
            
    def f_prime(x):
        return some_dae.jacobian_encode(x)

    return {'r':r, 'r_prime':r_prime,
            'f':f, 'f_prime':f_prime}


if __name__ == "__main__":
    main(sys.argv)
