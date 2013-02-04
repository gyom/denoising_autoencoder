

import time, os, sys, getopt
import numpy as np

import metropolis_hastings_sampler

def usage():
    print "-- usage example --"
    print "python script_1.py --n_samples=100 --n_chains=1000 --thinning_factor=100 --langevin_lambda=0.01 --mcmc_method=metropolis_hastings_langevin_E --dataset=ninja_star"

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["help", "n_samples=", "n_chains=", "thinning_factor=", "burn_in=", "langevin_lambda=", "mcmc_method=", "proposal_stddev=", "dataset="])
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

    #{'samples': samples_for_all_chains,
    # 'elapsed_time':sampling_end_time - sampling_start_time,
    # 'proposals_per_second':proposals_per_second,
    # 'acceptance_ratio':combined_acceptance_ratio    }

    print "Got the samples. Acceptance ratio was %f" % results['acceptance_ratio']
    print "MCMC proposal speed was 10^%0.2f / s" % (np.log(results['proposals_per_second']) / np.log(10), )

    if len(results['samples'].shape) == 2:
        cross_entropy = output_options['cross_entropy_function'](results['samples'])
        print "The cross-entropy of the samples is %f. Smaller values are best." % cross_entropy
    elif len(results['samples'].shape) == 3:
        cross_entropy = output_options['cross_entropy_function'](results['samples'][:,-1,:])
        print "The cross-entropy of the samples for all chains at the last time step is %f. Smaller values are best." % cross_entropy
    else:
        error("Wrong shape for samples returned !")



# Implement metropolis_hastings_sampler.run_chain_with_energy
# with grad_E instead.

if False:
    import matplotlib
    matplotlib.use('Agg')
    import pylab
    import matplotlib.pyplot as plt

    pylab.hold(True)
    pylab.scatter(X[:,0], X[:,1])

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
    id_number = int(np.random.uniform(0,10000))
    outfile = "/u/alaingui/umontreal/metropolis_hastings_langevin/script_1_%s_%0.5d.png" % (mcmc_method, id_number)
    pylab.savefig(outfile, dpi=100)
    pylab.close()
    print "Wrote %s" % (outfile,)

    quit()



if __name__ == "__main__":
    main()