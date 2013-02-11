

import time, os, sys, getopt
import numpy as np

import metropolis_hastings_sampler

def usage():
    print "-- usage example --"
    print "python script_1.py --n_samples=100 --n_chains=1000 --thinning_factor=100 --langevin_lambda=0.01 --mcmc_method=metropolis_hastings_langevin_E --dataset=ninja_star"

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["help", "n_samples=", "n_chains=", "thinning_factor=", "burn_in=", "langevin_lambda=", "mcmc_method=", "proposal_stddev=", "dataset_desc=", "output_dir_prefix="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    sampling_options = {}
    sampling_options["n_chains"] = None

    output_options = {}

    verbose = False
    for o, a in opts:
        if o == "-v":
            # unused
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--n_samples"):
            sampling_options["n_samples"] = int(a)
        elif o in ("--thinning_factor"):
            sampling_options["thinning_factor"] = int(a)
        elif o in ("--burn_in"):
            sampling_options["burn_in"] = int(a)
        elif o in ("--langevin_lambda"):
            sampling_options["langevin_lambda"] = float(a)
        elif o in ("--proposal_stddev"):
            sampling_options["proposal_stddev"] = float(a)
        elif o in ("--n_chains"):
            sampling_options["n_chains"] = int(a)
        elif o in ("--mcmc_method"):
            sampling_options["mcmc_method"] = a
            #if not a in ['langevin',
            #             'metropolis_hastings_langevin_E',
            #             'metropolis_hastings_langevin_grad_E',
            #             'metropolis_hastings_E',
            #             'metropolis_hastings_grad_E']:
            #    error("Bad name for mcmc_method.")
        elif o in ("--dataset_desc"):
            if a in ['ninja_star', 'Salah_DAE', 'guillaume_DAE']:
                sampling_options["dataset_desc"] = a
            else:
                "Unrecognized dataset."
        elif o in ("--output_dir_prefix"):
            output_options['output_dir_prefix'] = a
        else:
            assert False, "unhandled option"


    if sampling_options["dataset_desc"] == "ninja_star":

        import ninja_star_distribution
        sampling_options["E"] = ninja_star_distribution.E
        sampling_options["grad_E"] = ninja_star_distribution.grad_E

        if not sampling_options["n_chains"] == None:
            sampling_options["x0"] = np.random.normal(size=(sampling_options["n_chains"],2))
        else:
            sampling_options["x0"] = np.random.normal(size=(2,))

        # this field should be obsolete
        #output_options["cross_entropy_function"] = ninja_star_distribution.cross_entropy

    elif sampling_options["dataset_desc"] == "butterfly":

        import butterfly_distribution
        sampling_options["E"] = butterfly_distribution.E
        sampling_options["grad_E"] = butterfly_distribution.grad_E

        if not sampling_options["n_chains"] == None:
            sampling_options["x0"] = np.random.normal(size=(sampling_options["n_chains"],2))
        else:
            sampling_options["x0"] = np.random.normal(size=(2,))        

    elif sampling_options["dataset_desc"] == "Salah_DAE":

        import cPickle
        salah_dae_params = cPickle.load(open('/data/lisatmp2/rifaisal/share/sm_dae/ae_std=0.4.pkl'))
 
        import dae_untied_weights
        n_inputs = 784
        n_hiddens = 1024
        Wc = salah_dae_params['e_weights']
        Wb = salah_dae_params['e_weights']
        c = salah_dae_params['e_bias']
        b = salah_dae_params['d_bias']
        s = 1.0/16

        the_dae = dae_untied_weights.DAE_untied_weights(
            n_inputs=n_inputs,
            n_hiddens=n_hiddens,
            Wc=Wc, Wb=Wb,
            c=c, b=b,
            s=s, act_func=['sigmoid', 'id'])
        reference_langevin_lambda = 0.4**2
        # overwrite whichever langevin lambda was given as argument
        #sampling_options["langevin_lambda"] = reference_langevin_lambda
        r = lambda x: the_dae.encode_decode(x.reshape((-1,n_inputs))).reshape((n_inputs,))
        grad_E = lambda x: - (r(x) - x) / reference_langevin_lambda
        sampling_options["grad_E"] = grad_E
 
        # METHOD 1
        mnist_dataset = cPickle.load(open('/data/lisa/data/mnist/mnist.pkl', 'rb'))
        mnist_train = mnist_dataset[0]
        mnist_train_data, mnist_train_labels = mnist_train
        ind = np.random.randint(0, mnist_train_data.shape[0]-1)
        sampling_options["x0"] = mnist_train_data[ind,:]
        print "Starting the simulation from MNIST digit %d" % mnist_train_labels[ind]
        # METHOD 3
        #sampling_options["x0"] = np.random.uniform(low=0.0, high=1.0,size=(n_inputs,))

        # these are the only valid sampling methods because they rely only on grad_E
        assert sampling_options["mcmc_method"] in ["langevin", "metropolis_hastings_grad_E", "metropolis_hastings_langevin_grad_E"]

    elif sampling_options["dataset_desc"] == "guillaume_DAE":

        import cPickle
        #guillaume_dae_params = cPickle.load(open("/u/alaingui/umontreal/denoising_autoencoder/mcmc_pof/trained_models/mydae_2013_02_07.pkl"))
        guillaume_dae_params = cPickle.load(open("/u/alaingui/umontreal/denoising_autoencoder/mcmc_pof/trained_models/mydae_2013_02_08.pkl"))

        import dae_untied_weights
        n_inputs = 784
        n_hiddens = 1024
        Wc = guillaume_dae_params['Wc']
        Wb = guillaume_dae_params['Wb']
        c = guillaume_dae_params['c']
        b = guillaume_dae_params['b']
        s = guillaume_dae_params['s']
        act_func = guillaume_dae_params['act_func']

        the_dae = dae_untied_weights.DAE_untied_weights(
            n_inputs=n_inputs,
            n_hiddens=n_hiddens,
            Wc=Wc, Wb=Wb,
            c=c, b=b,
            s=s, act_func=act_func)

        reference_langevin_lambda = 0.01
        #reference_langevin_lambda = 1.0
        # overwrite whichever langevin lambda was given as argument
        #sampling_options["langevin_lambda"] = reference_langevin_lambda
        r = lambda x: the_dae.encode_decode(x.reshape((-1,n_inputs))).reshape((n_inputs,))
        grad_E = lambda x: - (r(x) - x) / reference_langevin_lambda
        sampling_options["grad_E"] = grad_E

        # METHOD 1
        mnist_dataset = cPickle.load(open('/data/lisa/data/mnist/mnist.pkl', 'rb'))
        mnist_train = mnist_dataset[0]
        mnist_train_data, mnist_train_labels = mnist_train
        ind = np.random.randint(0, mnist_train_data.shape[0]-1)
        sampling_options["x0"] = mnist_train_data[ind,:]
        print "Starting the simulation from MNIST digit %d" % mnist_train_labels[ind]
        # METHOD 3
        #sampling_options["x0"] = np.random.uniform(low=0.0, high=1.0,size=(n_inputs,))

        # these are the only valid sampling methods because they rely only on grad_E
        assert sampling_options["mcmc_method"] in ["langevin", "metropolis_hastings_grad_E", "metropolis_hastings_langevin_grad_E"]
        

    else:
        error("No dataset was supplied.")


    results = metropolis_hastings_sampler.mcmc_generate_samples(sampling_options)
    #
    # Returns a dictionary of this form.
    #
    # {'samples': numpy array (n_chains, n_samples, d),
    #  'elapsed_time': seconds as float,
    #  'proposals_per_second': float,
    #  'acceptance_ratio': float in [0.0,1.0] }

    print "Got the samples. Acceptance ratio was %f" % results['acceptance_ratio']
    print "MCMC proposal speed was 10^%0.2f / s" % (np.log(results['proposals_per_second']) / np.log(10), )


    output_image_dir = "%s/%s/%d" % (output_options["output_dir_prefix"], sampling_options['mcmc_method'], int(time.time()) )
    os.makedirs(output_image_dir)

    import cPickle
    output_pkl_name = os.path.join(output_image_dir, "results_and_params.pkl")
    if sampling_options.has_key("grad_E"):
        sampling_options["grad_E"] = "CANNOT BE PICKLED"
    f = open(output_pkl_name, "w")
    cPickle.dump({'results':results, 'sampling_options':sampling_options, 'output_options':output_options}, f)
    f.close()
    print "Wrote " + output_pkl_name

    samples_only_pkl_name = os.path.join(output_image_dir, "samples.pkl")
    f = open(samples_only_pkl_name, "w")
    cPickle.dump(results['samples'], f)
    f.close()
    print "Wrote " + samples_only_pkl_name

    if sampling_options["dataset_desc"] in ["Salah_DAE", "guillaume_DAE"]:
        digits_image_file = os.path.join(output_image_dir, "digits.png")
        plot_Salah_DAE_samples(results['samples'],  digits_image_file)




def plot_Salah_DAE_samples(samples, image_output_file):

    import yann_dauphin_utils
    import PIL

    #image_output_file = "/u/alaingui/Documents/tmp/Salah_DAE_2013_02_06/metropolis_hastings_langevin_grad_E/1360188911/digits.png"
    #samples_pkl_file = "/u/alaingui/Documents/tmp/Salah_DAE_2013_02_06/metropolis_hastings_langevin_grad_E/1360188911/samples.pkl"
    #samples = cPickle.load(open(samples_pkl_file,"r"))

    assert len(samples.shape) == 2
    N = samples.shape[0]
    n_inputs = samples.shape[1]

    tile_j = int(np.ceil(np.sqrt(N)))
    tile_i = int(np.ceil(float(N) / tile_j))

    img_j = int(np.ceil(np.sqrt(n_inputs)))
    img_i = int(np.ceil(float(n_inputs) / img_j))

    from PIL import Image
    image = Image.fromarray(yann_dauphin_utils.tile_raster_images(
        X = samples,
        img_shape = (img_j,img_i), tile_shape = (tile_j, tile_i),
        tile_spacing=(1,1)))

    image.save(image_output_file)



if __name__ == "__main__":
    main()