
import os
import re
import numpy as np

KL_values = {}

methods = ["metropolis_hastings_E",
           "metropolis_hastings_grad_E",
           "langevin",
           "metropolis_hastings_langevin_E",
           "metropolis_hastings_langevin_grad_E"]

for method in methods:

    file = os.path.join("/u/alaingui/umontreal/denoising_autoencoder/mcmc_pof/run_once/ninja_star_checking_KL", "2013_02_12_KL_measurements_ninja_star_%s.txt" % (method,))
    assert os.path.exists(file)

    KL_values[method] = []
    for line in open(file, "r"):
        m = re.search(r"We got a KL divergence value of ([\d\.]+)", line)
        if m:
            #print m.group(1)
            #print m.groups(1)
            KL_value = float(m.group(1))
            #print KL_value
            KL_values[method].append(KL_value)

    KL_values[method] = np.array(KL_values[method])

for method in methods:

    print "method : %s" % method
    print KL_values[method]
    print "mean : %f" % KL_values[method].mean()
    print "   stddev : %f" % KL_values[method].std()