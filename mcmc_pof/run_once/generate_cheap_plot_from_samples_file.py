
import cPickle
import sys, os

assert len(sys.argv) == 2

input_file = sys.argv[1]
assert os.path.exists(input_file)

samples = cPickle.load(open(input_file, "r"))
print "Read " + input_file


import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt

output_file = os.path.join(os.path.dirname(input_file), "quick_plot.png")
pylab.hold(True)

pylab.scatter(samples[:,0], samples[:,1])

#M = 10
pylab.draw()
#pylab.axes([-M, M, -M, M])
pylab.savefig(output_file, dpi=100)
pylab.close()

print "Wrote " + output_file


# find /u/alaingui/Documents/tmp/2013_02_10_KL_measurements_butterfly/metropolis_hastings_E -name samples.pkl -exec python run_once/generate_cheap_plot_from_samples_file.py {} \;
# find /u/alaingui/Documents/tmp/2013_02_10_KL_measurements_butterfly/metropolis_hastings_grad_E -name samples.pkl -exec python run_once/generate_cheap_plot_from_samples_file.py {} \;
# find /u/alaingui/Documents/tmp/2013_02_10_KL_measurements_butterfly/metropolis_hastings_langevin_E -name samples.pkl -exec python run_once/generate_cheap_plot_from_samples_file.py {} \;
# find /u/alaingui/Documents/tmp/2013_02_10_KL_measurements_butterfly/metropolis_hastings_langevin_grad_E -name samples.pkl -exec python run_once/generate_cheap_plot_from_samples_file.py {} \;
# find /u/alaingui/Documents/tmp/2013_02_10_KL_measurements_butterfly/langevin -name samples.pkl -exec python run_once/generate_cheap_plot_from_samples_file.py {} \;

