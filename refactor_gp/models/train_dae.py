#!/bin/env python

import numpy as np
import os, sys, time

def usage():
    print "-- python train_dae.py --n_hiddens= --maxiter=50 --lbfgs_rank=20 --act_func='[\"tanh\", \"sigmoid\"]' --noise_stddevs='[1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001]' --train_samples_pickle=\"/u/alaingui/Documents/blah/3493/train_samples.pkl\" --valid_samples_pickle=\"/u/alaingui/Documents/blah/3493/valid_samples.pkl\" --output_dir=/u/alaingui/Documents/tmp/trained_dae_294343"
    print ""
    print "Some of the weirdness in the syntax is explained by the fact that we're using json for certain parameters."

def main(argv):
    """
       maxiter
       lbfgs_rank
       act_func
       noise_stddevs
       train_samples_pickle
       output_dir is the directory in which we'll write the results
    """

    import getopt
    import cPickle
    import json

    try:
        opts, args = getopt.getopt(argv[1:], "hv", ["n_hiddens=", "maxiter=", "lbfgs_rank=", "act_func=", "noise_stddevs=", "alt_valid_noise_stddevs=", "train_samples_pickle=", "valid_samples_pickle=", "test_samples_pickle=", "output_dir=", "want_early_termination=", "save_hidden_units=", "resume_dae_pickle="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    n_hiddens = None
    maxiter = 50
    lbfgs_rank = None
    act_func = None
    noise_stddevs = None
    alt_valid_noise_stddevs = None
    train_samples_pickle = None
    valid_samples_pickle = None
    test_samples_pickle = None
    output_dir = None
    want_early_termination = False
    # used for multilayers
    save_hidden_units = False
    resume_dae_pickle = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--n_hiddens"):
            n_hiddens = int(a)
        elif o in ("--maxiter"):
            maxiter = int(a)
        elif o in ("--lbfgs_rank"):
            lbfgs_rank = int(a)
        elif o in ("--act_func"):
            act_func = json.loads(a)
            assert type(act_func) == list
            assert len(act_func) == 2
        elif o in ("--noise_stddevs"):
            noise_stddevs = json.loads(a)
            assert type(noise_stddevs) in [list, float, int]
            if type(noise_stddevs) in [float, int]:
                noise_stddevs = [noise_stddevs]
        elif o in ("--alt_valid_noise_stddevs"):
            alt_valid_noise_stddevs = json.loads(a)
            assert type(alt_valid_noise_stddevs) in [list, float, int]
            if type(alt_valid_noise_stddevs) in [float, int]:
                alt_valid_noise_stddevs = [alt_valid_noise_stddevs]
        elif o in ("--train_samples_pickle"):
            train_samples_pickle = a
        elif o in ("--valid_samples_pickle"):
            valid_samples_pickle = a
        elif o in ("--test_samples_pickle"):
            test_samples_pickle = a
        elif o in ("--output_dir"):
            output_dir = a
        elif o in ("--want_early_termination"):
            want_early_termination = ((a == "True") or (a == "true") or (a=="1"))
        elif o in ("--save_hidden_units"):
            save_hidden_units = ((a == "True") or (a == "true") or (a=="1"))
        elif o in ("--resume_dae_pickle"):
            resume_dae_pickle = a
        else:
            assert False, "unhandled option"
 
    #print "want_early_termination is %d" % want_early_termination

    assert os.path.exists(train_samples_pickle)
    train_samples = cPickle.load(open(train_samples_pickle, 'rb'))
    _, n_inputs = train_samples.shape

    if valid_samples_pickle:
        # it's alright to omit the validation set,
        # but if it's specified as argument, it has to be
        # a legitimate pickle
        assert os.path.exists(valid_samples_pickle)
        valid_samples = cPickle.load(open(valid_samples_pickle, 'rb'))
        assert train_samples.shape[1] == valid_samples.shape[1]
    else:
        valid_samples = None


    from dae_untied_weights import DAE_untied_weights

    if resume_dae_pickle is not None:
        assert os.path.exists(resume_dae_pickle)
        mydae = DAE_untied_weights(dae_pickle_file = resume_dae_pickle)
        if n_hiddens is not None:
            assert mydae.n_hiddens == n_hiddens
            assert mydae.n_inputs  == n_inputs
    else:
        mydae = DAE_untied_weights(n_inputs = n_inputs,
                                   n_hiddens = n_hiddens,
                                   act_func = act_func)

    #mydae.fit_with_decreasing_noise(mnist_train_data[0:2000,:],
    #                                [0.1, 0.05, 0.01, 0.001],
    #                                {'method' : 'fmin_cg',
    #                                 'maxiter' : 500,
    #                                 'gtol':0.001})

    start_time = time.time()

    if want_early_termination:
        early_termination_args = {'stop_if_loss_greater_than':"auto"}
    else:
        early_termination_args = {}

    (train_model_losses, valid_model_losses, post_valid_model_losses, post_alt_valid_model_losses) = \
        mydae.fit_with_decreasing_noise(train_samples,
                                        noise_stddevs,
                                        {'method' : 'fmin_l_bfgs_b',
                                         'maxiter' : maxiter,
                                         'm':lbfgs_rank},
                                        early_termination_args,
                                        X_valid = valid_samples,
                                        list_of_additional_valid_stddev = alt_valid_noise_stddevs)
    #print "post_valid_model_losses"
    #print post_valid_model_losses
    #print

    # We'll simplify some of the later steps by simply
    # referring to "model_losses" for the version that
    # was used. This will be used later when loading
    # the model in sample_dae.py, but it's not actually
    # necessary since we could also reason things out
    # from the np.nan present.
    if not (valid_samples == None):
        model_losses = valid_model_losses
    else:
        model_losses = train_model_losses

    end_time = time.time()
    computational_cost_in_seconds = int(end_time - start_time)
    print "Training took %d seconds." % (computational_cost_in_seconds,)

    early_termination_occurred = False
    if np.nan in model_losses:
        early_termination_occurred = True
        print "We terminated early in the training because we couldn't do better than the identity function r(x) = x."

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print "Creating directory %s" % (output_dir,)

    trained_model_pickle_file = os.path.join(output_dir, "trained_dae.pkl")
    mydae.save_pickle(trained_model_pickle_file)
    print "Wrote trained DAE in %s" % (trained_model_pickle_file,)

    if test_samples_pickle:
        assert os.path.exists(test_samples_pickle)
        test_samples = cPickle.load(open(test_samples_pickle, 'rb'))
        assert train_samples.shape[1] == test_samples.shape[1]
    else:
        test_samples = None


    extra_pickle_file = os.path.join(output_dir, "extra_details.pkl")
    extra_json_file = os.path.join(output_dir, "extra_details.json")
    extra_details = {'n_inputs':n_inputs,
                     'n_hiddens':n_hiddens,
                     'maxiter':maxiter,
                     'lbfgs_rank':lbfgs_rank,
                     'act_func':act_func,
                     'noise_stddevs':noise_stddevs,
                     'train_samples_pickle':train_samples_pickle,
                     'valid_samples_pickle':valid_samples_pickle,
                     'test_samples_pickle':test_samples_pickle,
                     'model_losses':model_losses,
                     'train_model_losses':train_model_losses,
                     'valid_model_losses':valid_model_losses,
                     'post_valid_model_losses':post_valid_model_losses,
                     'post_alt_valid_model_losses':post_alt_valid_model_losses,
                     'computational_cost_in_seconds':computational_cost_in_seconds,
                     'early_termination_occurred':early_termination_occurred}

    if save_hidden_units:
        for (samples, prefix) in [(train_samples, 'train'),
                                  (valid_samples, 'valid'),
                                  (test_samples,  'test')]:
            if samples is not None:
                hidden_units = mydae.encode(samples)
                hidden_units_pickle_file = os.path.join(output_dir, "%s_hidden_units.pkl" % (prefix,))
                extra_details['%s_hidden_units_pickle' % (prefix,)] = hidden_units_pickle_file
                cPickle.dump(hidden_units, open(hidden_units_pickle_file, "w"))
                print "Wrote %s" % (hidden_units_pickle_file,)
                del hidden_units
                del hidden_units_pickle_file

    cPickle.dump(extra_details, open(extra_pickle_file, "w"))
    json.dump(extra_details, open(extra_json_file, "w"), sort_keys=True, indent=4, separators=(',', ': '))
    print "Wrote %s" % (extra_pickle_file,)
    print "Wrote %s" % (extra_json_file,)


if __name__ == "__main__":
    main(sys.argv)
