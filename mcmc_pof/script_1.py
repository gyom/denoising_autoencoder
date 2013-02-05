

import time, os, sys, getopt
import numpy as np

import metropolis_hastings_sampler

def usage():
    print "-- usage example --"
    print "python script_1.py --n_samples=100 --n_chains=1000 --thinning_factor=100 --langevin_lambda=0.01 --mcmc_method=metropolis_hastings_langevin_E --dataset=ninja_star"

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["help", "n_samples=", "n_chains=", "thinning_factor=", "burn_in=", "langevin_lambda=", "mcmc_method=", "proposal_stddev=", "dataset=", "output_dir_prefix=", "reference_pickled_samples_for_KL=", "reference_stddev_for_KL="])
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
        elif o in ("--dataset"):
            if a == 'ninja_star':
                import ninja_star_distribution
                sampling_options["dataset_description"] = "ninja_star"
                sampling_options["E"] = ninja_star_distribution.E
                sampling_options["grad_E"] = ninja_star_distribution.grad_E
            else:
                "Unrecognized dataset."
        elif o in ("--output_dir_prefix"):
            output_options['output_dir_prefix'] = a
        elif o in ("--reference_pickled_samples_for_KL"):
            output_options['reference_pickled_samples_for_KL'] = a
        elif o in ("--reference_stddev_for_KL"):
            output_options['reference_stddev_for_KL'] = float(a)
        else:
            assert False, "unhandled option"

    if sampling_options["dataset_description"] == "ninja_star":
        if not sampling_options["n_chains"] == None:
            sampling_options["x0"] = np.random.normal(size=(sampling_options["n_chains"],2))
        else:
            sampling_options["x0"] = np.random.normal(size=(2,))
        
        output_options["cross_entropy_function"] = ninja_star_distribution.cross_entropy
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


    if len(results['samples'].shape) == 2:

        # We will evaluate an approximate KL divergence if we are given a
        # reference set of samples from the true distribution.
        if output_options.has_key("reference_pickled_samples_for_KL"):

            import KL_approximation
            import cPickle
            assert os.path.exists(output_options["reference_pickled_samples_for_KL"])
            f = open(output_options["reference_pickled_samples_for_KL"])
            reference_sample = cPickle.load(f)

            if output_options.has_key("reference_stddev_for_KL"):
                reference_stddev_for_KL = output_options["reference_stddev_for_KL"]
            else:
                reference_stddev_for_KL = None

            KL_value = KL_approximation.KL(reference_sample, results['samples'], 2000, 2000, stddev = reference_stddev_for_KL)
            print "We got a KL divergence value of %f" % KL_value


    #if len(results['samples'].shape) == 2:
    #    cross_entropy = output_options['cross_entropy_function'](results['samples'])
    #    print "The cross-entropy of the samples is %f. Smaller values are best." % cross_entropy
    #elif len(results['samples'].shape) == 3:
    #    cross_entropy = output_options['cross_entropy_function'](results['samples'][:,-1,:])
    #    print "The cross-entropy of the samples for all chains at the last time step is %f. Smaller values are best." % cross_entropy
    #else:
    #    raise("Wrong shape for samples returned !")
    
    # TODO : Have that directory be specified as an argument.
    #output_image_dir = "/u/alaingui/Documents/tmp/%s/%d" % ( sampling_options['mcmc_method'], int(time.time()) )
    output_image_dir = "%s/%s/%d" % (output_options["output_dir_prefix"], sampling_options['mcmc_method'], int(time.time()) )
    os.makedirs(output_image_dir)

    if len(results['samples'].shape) == 2:
    
        output_image_path = os.path.join(output_image_dir, "whole_chain.png")
        plot_one_slice_of_ninja_star_samples(results['samples'], output_image_path, dpi=200)
    
    elif len(results['samples'].shape) == 3:

        for n in np.arange(sampling_options['n_samples']):
            output_image_path = os.path.join(output_image_dir, "frame_%0.6d.png" % n)
            plot_one_slice_of_ninja_star_samples(results['samples'][:,n,:], output_image_path, dpi=100)

        import subprocess
        #export MOVIEDIR=${HOME}/Documents/tmp/metropolis_hastings_E/1359965409
        #ffmpeg -i ${MOVIEDIR}/frame_%06d.png -vcodec libx264 -vpre hq -crf 22 -r 10 ${MOVIEDIR}/seq.mp4
        subprocess.check_output(["ffmpeg", "-i", output_image_dir + r"/frame_%06d.png", "-y", "-vcodec", "libx264", "-vpre", "hq", "-crf", "22", "-r", "10", output_image_dir + r"/assembled.mp4"])
        print "Generated movie at %s." % (output_image_dir + r"/assembled.mp4", )
    else:
        raise("Wrong shape for samples returned !")
    

    




# We'll put the imports here just in case the
# global scope wouldn't be the best idea.
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt


def plot_one_slice_of_ninja_star_samples(samples_slice, output_image_path, dpi=100):
    """
        Samples should be of size (M, d).
        You would generally pick either one chain alone
        or one time slice from a set of chains.

        This plotting function doesn't pretend to be
        a very general method. It assumes that the samples
        are 2-dimensional and that [-4.0, 4.0]^2 is the
        best choice of window to plot the samples.
    """

    import ninja_star_distribution

    pylab.hold(True)

    x = samples_slice[:,0]
    y = samples_slice[:,1]

    # TODO : pick better color for the sample dots
    pylab.scatter(x, y)
    # TODO : stamp the KL divergence on the plots


    print "Computing the original pdf values."
    M = 4.0
    mesh_x,mesh_y = np.mgrid[-M:M:.01, -M:M:.01]
    z = ninja_star_distribution.mesh_pdf(mesh_x, mesh_y)

    print "Generating the nice plots."
    model_pdf_values_plot_handle = plt.pcolor(mesh_x, mesh_y, z)
    #plt.winter()
    plt.pink()
    #d = plt.colorbar(model_pdf_value_plot_handle, orientation='horizontal')

    pylab.draw()
    #pylab.savefig(output_image_path)
    pylab.savefig(output_image_path, dpi=dpi)
    pylab.close()




if __name__ == "__main__":
    main()