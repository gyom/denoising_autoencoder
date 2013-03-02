
import subprocess
import cPickle

# pseudo-parameter
#dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d10/experiment_09"
#dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d100/experiment_01"

dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d2/experiment_12"
list_files = sorted([e for e in subprocess.check_output("find %s -name extra_details.pkl" % (dir,), shell=True).split("\n") if len(e)>0])

dir = "/data/lisatmp2/alaingui/dae/generated_samples/experiment_11_03"
list_files = sorted([e for e in subprocess.check_output("find %s -name sampling_extra_details.pkl" % (dir,), shell=True).split("\n") if len(e)>0])


list_contents = [cPickle.load(open(f)) for f in list_files]

best_ind = [10, 11, 19, 9, 27, 28, 0, 37, 18, 1]
for i in best_ind:
    print list_contents[i]



#[e for e in list_contents if (e['maxiter'] == 1000) and (e['act_func'] == ['tanh', 'tanh'])]

A = np.array([min(*e['valid_model_losses']) for e in list_contents])
B = np.array([e['post_valid_model_losses'][10] for e in list_contents])

iB = np.argsort(B)
iA = np.argsort(A)

#np.vstack((A[iA], B[iA]))
np.hstack((A[iA].reshape((-1,1)), B[iA].reshape((-1,1))))
np.hstack((A[iB].reshape((-1,1)), B[iB].reshape((-1,1))))

# This should get us the best index.
# Manual inspection of A,B suggests this index.
# Visual inspected too.
# Now the log also suggests this.
np.argmin(np.log(A) + np.log(B))

#sorted(A)
#i = np.argmin(A)
#list_abs_path_files[i]


for key in ['n_hiddens', 'maxiter', 'act_func']:
    print ""
    print "--------------------------------------------------------"
    print key
    print np.array([list_contents[i][key] for i in np.argsort(A)])
    print "--------------------------------------------------------"


np.argsort(A)[0:10]
Out[134]: array([109,  15, 118,  85,  32,  35, 114, 116,  28,   7])


np.array([list_contents[i]['n_hiddens'] for i in np.argsort(A)])
