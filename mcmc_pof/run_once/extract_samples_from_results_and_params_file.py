
import cPickle
import sys, os

assert len(sys.argv) == 2

input_file = sys.argv[1]
assert os.path.exists(input_file)

results_and_params = cPickle.load(open(input_file, "r"))
assert results_and_params.has_key("results")
assert results_and_params["results"].has_key("samples")
print "Read " + input_file

samples = results_and_params["results"]["samples"]
output_file = os.path.join(os.path.dirname(input_file), "samples.pkl")
cPickle.dump(samples, open(output_file, "w"))
print "Wrote " + output_file


# find /u/alaingui/Documents/tmp/2013_02_09_KL_measurements -name results_and_params.pkl -exec python extract_samples_from_results_and_params_file.py {} \;