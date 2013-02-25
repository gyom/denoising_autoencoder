

import numpy as np
import os, sys, time

def usage():
    print "-- python train_dae.py --n_hiddens= --maxiter=50 --lbfgs_rank=20 --act_func='[\"tanh\", \"sigmoid\"]' --noise_stddevs='[1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001]' --train_samples_pickle=\"/u/alaingui/Documents/blah/3493/train_samples.pkl\" --output_dir=/u/alaingui/Documents/tmp/trained_dae_294343"
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
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["n_hiddens=", "maxiter=", "lbfgs_rank=", "act_func=", "noise_stddevs=", "train_samples_pickle=", "output_dir=", "want_early_termination="])
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
    train_samples_pickle = None
    output_dir = None
    want_early_termination = False

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
        elif o in ("--train_samples_pickle"):
            train_samples_pickle = a
        elif o in ("--output_dir"):
            output_dir = a
        elif o in ("--want_early_termination"):
            want_early_termination = ((a == "True") or (a == "true") or (a=="1"))
        else:
            assert False, "unhandled option"
 
    #print "want_early_termination is %d" % want_early_termination

    assert os.path.exists(train_samples_pickle)
    samples = cPickle.load(open(train_samples_pickle, 'rb'))
    n_samples, n_inputs = samples.shape


    from dae_untied_weights import DAE_untied_weights

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

    model_losses = mydae.fit_with_decreasing_noise(samples,
                                                   noise_stddevs,
                                                   {'method' : 'fmin_l_bfgs_b',
                                                    'maxiter' : maxiter,
                                                   'm':lbfgs_rank},
                                                   early_termination_args)
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


    extra_pickle_file = os.path.join(output_dir, "extra_details.pkl")
    extra_json_file = os.path.join(output_dir, "extra_details.json")
    extra_details = {'n_inputs':n_inputs,
                     'n_hiddens':n_hiddens,
                     'maxiter':maxiter,
                     'lbfgs_rank':lbfgs_rank,
                     'act_func':act_func,
                     'noise_stddevs':noise_stddevs,
                     'train_samples_pickle':train_samples_pickle,
                     'model_losses':model_losses,
                     'computational_cost_in_seconds':computational_cost_in_seconds,
                     'early_termination_occurred':early_termination_occurred}

    cPickle.dump(extra_details, open(extra_pickle_file, "w"))
    json.dump(extra_details, open(extra_json_file, "w"))
    print "Wrote %s" % (extra_pickle_file,)
    print "Wrote %s" % (extra_json_file,)


if __name__ == "__main__":
    main(sys.argv)
